use openmaths::Vector3;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::operations::triangulate::triangulate_polygon_with_holes;
use crate::spatial::bvh::{Aabb3, Bvh3, BvhPairTraversalStats, BvhPrimitive};
use crate::spatial::placement::Placement3D;

use super::{Brep, CurveGeometry, SurfaceGeometry};

const METHOD: &str = "brep-triangle-boundary-proximity";
const PAIR_SCHEMA: &str = "opengeometry-brep-proximity-v1";
const POINT_SCHEMA: &str = "opengeometry-brep-point-proximity-v1";
const TOUCH_EPSILON_FACTOR: f64 = 1.0e-9;
const RAY_DIRECTION: Vector3 = Vector3 {
    x: 0.4915391523114243,
    y: 0.6121167066403734,
    z: 0.6198651932735114,
};

fn js_error(message: impl AsRef<str>) -> JsValue {
    #[cfg(target_arch = "wasm32")]
    {
        JsValue::from_str(message.as_ref())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = message;
        JsValue::NULL
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ProximityAccuracy {
    ExactPlanarBrep,
    TessellatedAnalyticSurfaces,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PairRelation {
    Separated,
    ZeroClearance,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PointRelation {
    Outside,
    InsideOrBoundary,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PairClassification {
    BoundaryDistance,
    BoundaryContactOrCrossing,
    Containment,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PointClassification {
    BoundaryDistance,
    BoundaryContact,
    Containment,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ProximitySource {
    BoundaryDistance,
    BoundaryContact,
    Containment,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ProximityAcceleration {
    TriangleBvhV1,
}

type JsonPoint = [f64; 3];

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BrepProximityReport {
    schema: String,
    method: String,
    accuracy: ProximityAccuracy,
    distance: f64,
    closest_point_lhs: JsonPoint,
    closest_point_rhs: JsonPoint,
    relation: PairRelation,
    classification: PairClassification,
    source: ProximitySource,
    lhs_triangle_count: usize,
    rhs_triangle_count: usize,
    acceleration: ProximityAcceleration,
    bvh_node_pair_tests: usize,
    triangle_pair_tests: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BrepPointProximityReport {
    schema: String,
    method: String,
    accuracy: ProximityAccuracy,
    distance: f64,
    query_point: JsonPoint,
    closest_point: JsonPoint,
    relation: PointRelation,
    classification: PointClassification,
    source: ProximitySource,
    triangle_count: usize,
}

#[derive(Clone)]
struct TriangleRecord {
    a: Vector3,
    b: Vector3,
    c: Vector3,
}

struct PreparedBrep {
    triangles: Vec<TriangleRecord>,
    triangle_bvh: Bvh3,
    vertices: Vec<Vector3>,
    bbox: BoundingBox,
    accuracy: ProximityAccuracy,
}

#[derive(Clone, Copy)]
struct BoundingBox {
    min: Vector3,
    max: Vector3,
}

#[derive(Clone, Copy)]
struct ContactWitness {
    lhs_point: Vector3,
    rhs_point: Vector3,
    distance_sq: f64,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PointJsonInput {
    Object { x: f64, y: f64, z: f64 },
    Array([f64; 3]),
}

#[derive(Deserialize)]
struct PlacementJsonInput {
    position: [f64; 3],
    rotation: [f64; 3],
    scale: [f64; 3],
}

#[wasm_bindgen(js_name = calculateBrepProximity)]
pub fn calculate_brep_proximity(lhs_json: String, rhs_json: String) -> Result<String, JsValue> {
    calculate_brep_proximity_with_placements(lhs_json, None, rhs_json, None)
}

#[wasm_bindgen(js_name = calculatePlacedBrepProximity)]
pub fn calculate_placed_brep_proximity(
    lhs_json: String,
    lhs_placement_json: String,
    rhs_json: String,
    rhs_placement_json: String,
) -> Result<String, JsValue> {
    calculate_brep_proximity_with_placements(
        lhs_json,
        Some(lhs_placement_json),
        rhs_json,
        Some(rhs_placement_json),
    )
}

#[wasm_bindgen(js_name = calculateBrepPointProximity)]
pub fn calculate_brep_point_proximity(
    brep_json: String,
    point_json: String,
) -> Result<String, JsValue> {
    calculate_brep_point_proximity_with_placement(brep_json, None, point_json)
}

#[wasm_bindgen(js_name = calculatePlacedBrepPointProximity)]
pub fn calculate_placed_brep_point_proximity(
    brep_json: String,
    placement_json: String,
    point_json: String,
) -> Result<String, JsValue> {
    calculate_brep_point_proximity_with_placement(brep_json, Some(placement_json), point_json)
}

fn calculate_brep_proximity_with_placements(
    lhs_json: String,
    lhs_placement_json: Option<String>,
    rhs_json: String,
    rhs_placement_json: Option<String>,
) -> Result<String, JsValue> {
    let lhs = load_prepared_brep(&lhs_json, lhs_placement_json.as_deref())?;
    let rhs = load_prepared_brep(&rhs_json, rhs_placement_json.as_deref())?;
    let report = compare_prepared_breps(&lhs, &rhs);
    ensure_finite_report_pair(&report)?;
    serde_json::to_string(&report)
        .map_err(|error| js_error(format!("Failed to serialize proximity report: {error}")))
}

fn calculate_brep_point_proximity_with_placement(
    brep_json: String,
    placement_json: Option<String>,
    point_json: String,
) -> Result<String, JsValue> {
    let prepared = load_prepared_brep(&brep_json, placement_json.as_deref())?;
    let point = parse_point_json(&point_json)?;
    if !point_is_finite(point) {
        return Err(js_error(
            "Point proximity input rejected: query point contains non-finite coordinates",
        ));
    }
    let report = compare_prepared_brep_to_point(&prepared, point);
    ensure_finite_report_point(&report)?;
    serde_json::to_string(&report)
        .map_err(|error| js_error(format!("Failed to serialize proximity report: {error}")))
}

fn load_prepared_brep(
    brep_json: &str,
    placement_json: Option<&str>,
) -> Result<PreparedBrep, JsValue> {
    let mut brep = parse_brep_json(brep_json)?;
    if let Some(placement_json) = placement_json {
        let placement = parse_placement_json(placement_json)?;
        brep = apply_placement(brep, &placement)?;
    }
    prepare_brep(&brep)
}

fn parse_brep_json(brep_json: &str) -> Result<Brep, JsValue> {
    serde_json::from_str::<Brep>(brep_json)
        .map_err(|error| js_error(format!("Invalid BRep JSON: {error}")))
}

fn parse_placement_json(placement_json: &str) -> Result<Placement3D, JsValue> {
    let placement_input = serde_json::from_str::<PlacementJsonInput>(placement_json)
        .map_err(|error| js_error(format!("Invalid placement JSON: {error}")))?;

    let position = Vector3::new(
        placement_input.position[0],
        placement_input.position[1],
        placement_input.position[2],
    );
    let rotation = Vector3::new(
        placement_input.rotation[0],
        placement_input.rotation[1],
        placement_input.rotation[2],
    );
    let scale = Vector3::new(
        placement_input.scale[0],
        placement_input.scale[1],
        placement_input.scale[2],
    );

    if !point_is_finite(position) || !point_is_finite(rotation) || !point_is_finite(scale) {
        return Err(js_error(
            "Placement proximity input rejected: placement contains non-finite coordinates",
        ));
    }

    let mut placement = Placement3D::new();
    placement
        .set_transform(position, rotation, scale)
        .map_err(|error| js_error(format!("Invalid placement scale: {error}")))?;
    Ok(placement)
}

fn apply_placement(brep: Brep, placement: &Placement3D) -> Result<Brep, JsValue> {
    let mut transformed = brep;
    transformed.apply_transform(placement);
    Ok(transformed)
}

fn prepare_brep(brep: &Brep) -> Result<PreparedBrep, JsValue> {
    brep.validate_topology()
        .map_err(|error| js_error(format!("BRep topology rejected for proximity: {error}")))?;

    if brep.faces.is_empty() || brep.shells.is_empty() {
        return Err(js_error(
            "BRep proximity rejected: closed solid shells are required",
        ));
    }

    if brep.shells.iter().any(|shell| !shell.is_closed) {
        return Err(js_error(
            "BRep proximity rejected: open shells are not supported",
        ));
    }

    ensure_brep_geometry_finite(brep)?;

    let mut triangles = Vec::new();
    for face in &brep.faces {
        let (face_vertices, holes_vertices) = brep.get_vertices_and_holes_by_face_id(face.id);
        if face_vertices.len() < 3 {
            return Err(js_error(format!(
                "BRep proximity rejected: face {} has fewer than three vertices",
                face.id
            )));
        }

        let face_triangles = triangulate_polygon_with_holes(&face_vertices, &holes_vertices);
        if face_triangles.is_empty() {
            return Err(js_error(format!(
                "BRep proximity rejected: face {} produced no valid triangles",
                face.id
            )));
        }

        let all_vertices: Vec<Vector3> = face_vertices
            .into_iter()
            .chain(holes_vertices.into_iter().flatten())
            .collect();

        for tri in face_triangles {
            let a = *all_vertices.get(tri[0]).ok_or_else(|| {
                js_error(format!(
                    "Invalid triangle index {} on face {}",
                    tri[0], face.id
                ))
            })?;
            let b = *all_vertices.get(tri[1]).ok_or_else(|| {
                js_error(format!(
                    "Invalid triangle index {} on face {}",
                    tri[1], face.id
                ))
            })?;
            let c = *all_vertices.get(tri[2]).ok_or_else(|| {
                js_error(format!(
                    "Invalid triangle index {} on face {}",
                    tri[2], face.id
                ))
            })?;

            if !triangle_is_finite(a, b, c) || triangle_area_sq(a, b, c) <= 1.0e-24 {
                return Err(js_error(format!(
                    "BRep proximity rejected: face {} contains non-finite or degenerate triangle",
                    face.id
                )));
            }

            triangles.push(TriangleRecord { a, b, c });
        }
    }

    if triangles.is_empty() {
        return Err(js_error(
            "BRep proximity rejected: no valid triangles were produced",
        ));
    }

    let vertices = brep
        .vertices
        .iter()
        .map(|vertex| vertex.position)
        .collect::<Vec<_>>();
    let bbox = BoundingBox::from_points(&vertices)
        .ok_or_else(|| js_error("BRep proximity rejected: no finite vertices were available"))?;
    let accuracy = if has_analytic_geometry(brep) {
        ProximityAccuracy::TessellatedAnalyticSurfaces
    } else {
        ProximityAccuracy::ExactPlanarBrep
    };
    let triangle_bvh = build_triangle_bvh(&triangles)?;

    Ok(PreparedBrep {
        triangles,
        triangle_bvh,
        vertices,
        bbox,
        accuracy,
    })
}

fn build_triangle_bvh(triangles: &[TriangleRecord]) -> Result<Bvh3, JsValue> {
    let primitives = triangles
        .iter()
        .enumerate()
        .map(|(index, triangle)| {
            let id = u32::try_from(index).map_err(|_| {
                js_error("BRep proximity rejected: triangle count exceeds spatial index capacity")
            })?;
            let min = Vector3::new(
                triangle.a.x.min(triangle.b.x).min(triangle.c.x),
                triangle.a.y.min(triangle.b.y).min(triangle.c.y),
                triangle.a.z.min(triangle.b.z).min(triangle.c.z),
            );
            let max = Vector3::new(
                triangle.a.x.max(triangle.b.x).max(triangle.c.x),
                triangle.a.y.max(triangle.b.y).max(triangle.c.y),
                triangle.a.z.max(triangle.b.z).max(triangle.c.z),
            );
            let bounds = Aabb3::new(min, max)
                .map_err(|error| js_error(format!("Invalid triangle bounds: {error}")))?;
            Ok(BvhPrimitive::new(id, bounds))
        })
        .collect::<Result<Vec<_>, JsValue>>()?;
    Ok(Bvh3::build(primitives))
}

fn ensure_brep_geometry_finite(brep: &Brep) -> Result<(), JsValue> {
    for vertex in &brep.vertices {
        if !point_is_finite(vertex.position) {
            return Err(js_error(format!(
                "BRep proximity rejected: vertex {} contains non-finite coordinates",
                vertex.id
            )));
        }
    }

    for edge in &brep.edges {
        if let Some(curve) = &edge.curve {
            if !curve_is_finite(curve) {
                return Err(js_error(format!(
                    "BRep proximity rejected: edge {} contains non-finite analytic geometry",
                    edge.id
                )));
            }
        }
    }

    for face in &brep.faces {
        if let Some(surface) = &face.surface {
            if !surface_is_finite(surface) {
                return Err(js_error(format!(
                    "BRep proximity rejected: face {} contains non-finite analytic geometry",
                    face.id
                )));
            }
        }
    }

    Ok(())
}

fn compare_prepared_breps(lhs: &PreparedBrep, rhs: &PreparedBrep) -> BrepProximityReport {
    let tolerance = lhs
        .bbox
        .diagonal_length()
        .max(rhs.bbox.diagonal_length())
        .max(1.0)
        * TOUCH_EPSILON_FACTOR;

    // One boundary traversal supplies both the contact decision and the
    // separated witness. Repeating the O(lhs_triangles * rhs_triangles)
    // traversal would double the dominant cost for ordinary separated solids.
    let (boundary_witness, traversal_stats) = triangle_distance(lhs, rhs);
    if boundary_witness.distance_sq <= tolerance * tolerance {
        return BrepProximityReport {
            schema: PAIR_SCHEMA.to_string(),
            method: METHOD.to_string(),
            accuracy: accuracy_for_pair(lhs.accuracy, rhs.accuracy),
            distance: 0.0,
            closest_point_lhs: point_to_json(boundary_witness.lhs_point),
            closest_point_rhs: point_to_json(boundary_witness.rhs_point),
            relation: PairRelation::ZeroClearance,
            classification: PairClassification::BoundaryContactOrCrossing,
            source: ProximitySource::BoundaryContact,
            lhs_triangle_count: lhs.triangles.len(),
            rhs_triangle_count: rhs.triangles.len(),
            acceleration: ProximityAcceleration::TriangleBvhV1,
            bvh_node_pair_tests: traversal_stats.node_pair_tests,
            triangle_pair_tests: traversal_stats.primitive_pair_tests,
        };
    }

    let rhs_inside_lhs = solid_contains_solid(lhs, rhs, tolerance);
    let lhs_inside_rhs = solid_contains_solid(rhs, lhs, tolerance);

    if rhs_inside_lhs || lhs_inside_rhs {
        let contained = if rhs_inside_lhs { rhs } else { lhs };
        let witness = containment_witness(contained, tolerance);
        return BrepProximityReport {
            schema: PAIR_SCHEMA.to_string(),
            method: METHOD.to_string(),
            accuracy: accuracy_for_pair(lhs.accuracy, rhs.accuracy),
            distance: 0.0,
            closest_point_lhs: point_to_json(witness),
            closest_point_rhs: point_to_json(witness),
            relation: PairRelation::ZeroClearance,
            classification: PairClassification::Containment,
            source: ProximitySource::Containment,
            lhs_triangle_count: lhs.triangles.len(),
            rhs_triangle_count: rhs.triangles.len(),
            acceleration: ProximityAcceleration::TriangleBvhV1,
            bvh_node_pair_tests: traversal_stats.node_pair_tests,
            triangle_pair_tests: traversal_stats.primitive_pair_tests,
        };
    }

    BrepProximityReport {
        schema: PAIR_SCHEMA.to_string(),
        method: METHOD.to_string(),
        accuracy: accuracy_for_pair(lhs.accuracy, rhs.accuracy),
        distance: boundary_witness.distance_sq.sqrt(),
        closest_point_lhs: point_to_json(boundary_witness.lhs_point),
        closest_point_rhs: point_to_json(boundary_witness.rhs_point),
        relation: PairRelation::Separated,
        classification: PairClassification::BoundaryDistance,
        source: ProximitySource::BoundaryDistance,
        lhs_triangle_count: lhs.triangles.len(),
        rhs_triangle_count: rhs.triangles.len(),
        acceleration: ProximityAcceleration::TriangleBvhV1,
        bvh_node_pair_tests: traversal_stats.node_pair_tests,
        triangle_pair_tests: traversal_stats.primitive_pair_tests,
    }
}

fn compare_prepared_brep_to_point(
    prepared: &PreparedBrep,
    point: Vector3,
) -> BrepPointProximityReport {
    let tolerance = prepared.bbox.diagonal_length().max(1.0) * TOUCH_EPSILON_FACTOR;
    let boundary_contact = point_to_triangles_distance(point, &prepared.triangles);

    if boundary_contact.distance_sq <= tolerance * tolerance {
        return BrepPointProximityReport {
            schema: POINT_SCHEMA.to_string(),
            method: METHOD.to_string(),
            accuracy: prepared.accuracy,
            distance: 0.0,
            query_point: point_to_json(point),
            closest_point: point_to_json(boundary_contact.rhs_point),
            relation: PointRelation::InsideOrBoundary,
            classification: PointClassification::BoundaryContact,
            source: ProximitySource::BoundaryContact,
            triangle_count: prepared.triangles.len(),
        };
    }

    if point_in_solid(prepared, point, tolerance) {
        return BrepPointProximityReport {
            schema: POINT_SCHEMA.to_string(),
            method: METHOD.to_string(),
            accuracy: prepared.accuracy,
            distance: 0.0,
            query_point: point_to_json(point),
            closest_point: point_to_json(point),
            relation: PointRelation::InsideOrBoundary,
            classification: PointClassification::Containment,
            source: ProximitySource::Containment,
            triangle_count: prepared.triangles.len(),
        };
    }

    BrepPointProximityReport {
        schema: POINT_SCHEMA.to_string(),
        method: METHOD.to_string(),
        accuracy: prepared.accuracy,
        distance: boundary_contact.distance_sq.sqrt(),
        query_point: point_to_json(point),
        closest_point: point_to_json(boundary_contact.rhs_point),
        relation: PointRelation::Outside,
        classification: PointClassification::BoundaryDistance,
        source: ProximitySource::BoundaryDistance,
        triangle_count: prepared.triangles.len(),
    }
}

fn triangle_distance(
    lhs: &PreparedBrep,
    rhs: &PreparedBrep,
) -> (ContactWitness, BvhPairTraversalStats) {
    let mut best = ContactWitness {
        lhs_point: lhs
            .vertices
            .first()
            .copied()
            .unwrap_or_else(|| Vector3::new(0.0, 0.0, 0.0)),
        rhs_point: rhs
            .vertices
            .first()
            .copied()
            .unwrap_or_else(|| Vector3::new(0.0, 0.0, 0.0)),
        distance_sq: f64::INFINITY,
    };

    let (_, stats) =
        lhs.triangle_bvh
            .visit_nearest_pairs(&rhs.triangle_bvh, f64::INFINITY, |lhs_id, rhs_id| {
                let candidate = triangle_pair_distance(
                    &lhs.triangles[lhs_id as usize],
                    &rhs.triangles[rhs_id as usize],
                    &lhs.bbox,
                    &rhs.bbox,
                );
                update_best(&mut best, candidate);
                candidate.distance_sq
            });

    (best, stats)
}

fn triangle_pair_distance(
    tri_lhs: &TriangleRecord,
    tri_rhs: &TriangleRecord,
    lhs_bbox: &BoundingBox,
    rhs_bbox: &BoundingBox,
) -> ContactWitness {
    if let Some(witness) = triangle_touch_witness(tri_lhs, tri_rhs, lhs_bbox, rhs_bbox) {
        return witness;
    }

    let mut best = ContactWitness {
        lhs_point: tri_lhs.a,
        rhs_point: tri_rhs.a,
        distance_sq: f64::INFINITY,
    };
    for point in [tri_lhs.a, tri_lhs.b, tri_lhs.c] {
        let closest = closest_point_on_triangle(point, tri_rhs);
        update_best(
            &mut best,
            ContactWitness {
                lhs_point: point,
                rhs_point: closest,
                distance_sq: distance_sq(point, closest),
            },
        );
    }
    for point in [tri_rhs.a, tri_rhs.b, tri_rhs.c] {
        let closest = closest_point_on_triangle(point, tri_lhs);
        update_best(
            &mut best,
            ContactWitness {
                lhs_point: closest,
                rhs_point: point,
                distance_sq: distance_sq(closest, point),
            },
        );
    }

    let lhs_edges = [
        (tri_lhs.a, tri_lhs.b),
        (tri_lhs.b, tri_lhs.c),
        (tri_lhs.c, tri_lhs.a),
    ];
    let rhs_edges = [
        (tri_rhs.a, tri_rhs.b),
        (tri_rhs.b, tri_rhs.c),
        (tri_rhs.c, tri_rhs.a),
    ];
    for (lhs_start, lhs_end) in lhs_edges {
        for (rhs_start, rhs_end) in rhs_edges {
            let (lhs_point, rhs_point, distance_sq) =
                closest_points_on_segments(lhs_start, lhs_end, rhs_start, rhs_end);
            update_best(
                &mut best,
                ContactWitness {
                    lhs_point,
                    rhs_point,
                    distance_sq,
                },
            );
        }
    }
    best
}

fn triangle_touch_witness(
    lhs: &TriangleRecord,
    rhs: &TriangleRecord,
    lhs_bbox: &BoundingBox,
    rhs_bbox: &BoundingBox,
) -> Option<ContactWitness> {
    for point in [lhs.a, lhs.b, lhs.c] {
        let closest = closest_point_on_triangle(point, rhs);
        if distance_sq(point, closest) <= touch_epsilon_sq(lhs_bbox, rhs_bbox) {
            return Some(ContactWitness {
                lhs_point: point,
                rhs_point: closest,
                distance_sq: distance_sq(point, closest),
            });
        }
    }

    for point in [rhs.a, rhs.b, rhs.c] {
        let closest = closest_point_on_triangle(point, lhs);
        if distance_sq(point, closest) <= touch_epsilon_sq(lhs_bbox, rhs_bbox) {
            return Some(ContactWitness {
                lhs_point: closest,
                rhs_point: point,
                distance_sq: distance_sq(closest, point),
            });
        }
    }

    let lhs_edges = [(lhs.a, lhs.b), (lhs.b, lhs.c), (lhs.c, lhs.a)];
    let rhs_edges = [(rhs.a, rhs.b), (rhs.b, rhs.c), (rhs.c, rhs.a)];
    for (lhs_start, lhs_end) in lhs_edges {
        if let Some(hit) = segment_triangle_intersection(lhs_start, lhs_end, rhs) {
            return Some(ContactWitness {
                lhs_point: hit,
                rhs_point: hit,
                distance_sq: 0.0,
            });
        }
    }
    for (rhs_start, rhs_end) in rhs_edges {
        if let Some(hit) = segment_triangle_intersection(rhs_start, rhs_end, lhs) {
            return Some(ContactWitness {
                lhs_point: hit,
                rhs_point: hit,
                distance_sq: 0.0,
            });
        }
    }

    if triangles_are_coplanar(lhs, rhs, lhs_bbox, rhs_bbox) && coplanar_triangles_overlap(lhs, rhs)
    {
        let witness = coplanar_overlap_witness(lhs, rhs);
        return Some(ContactWitness {
            lhs_point: witness,
            rhs_point: witness,
            distance_sq: 0.0,
        });
    }

    None
}

fn point_to_triangles_distance(point: Vector3, triangles: &[TriangleRecord]) -> ContactWitness {
    let mut best = ContactWitness {
        lhs_point: point,
        rhs_point: point,
        distance_sq: f64::INFINITY,
    };

    for triangle in triangles {
        let closest = closest_point_on_triangle(point, triangle);
        update_best(
            &mut best,
            ContactWitness {
                lhs_point: point,
                rhs_point: closest,
                distance_sq: distance_sq(point, closest),
            },
        );
    }

    best
}

fn containment_witness(prepared: &PreparedBrep, tolerance: f64) -> Vector3 {
    let center = prepared.bbox.center();
    if point_in_solid(prepared, center, tolerance) {
        return center;
    }

    prepared
        .vertices
        .first()
        .copied()
        .unwrap_or_else(|| Vector3::new(0.0, 0.0, 0.0))
}

fn solid_contains_solid(container: &PreparedBrep, inner: &PreparedBrep, tolerance: f64) -> bool {
    if inner.vertices.is_empty() {
        return false;
    }

    inner
        .vertices
        .iter()
        .copied()
        .all(|point| point_in_solid(container, point, tolerance))
}

fn point_in_solid(prepared: &PreparedBrep, point: Vector3, tolerance: f64) -> bool {
    if point_on_boundary(prepared, point, tolerance) {
        return true;
    }

    let mut hits = Vec::new();
    for triangle in &prepared.triangles {
        if let Some(t) = ray_triangle_intersection(point, RAY_DIRECTION, triangle, tolerance) {
            if t > tolerance {
                hits.push(t);
            }
        }
    }

    hits.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    let mut unique_hits = 0_usize;
    let mut last_hit: Option<f64> = None;
    for hit in hits {
        if let Some(previous) = last_hit {
            if (hit - previous).abs() <= tolerance.max(1.0e-12) {
                continue;
            }
        }
        unique_hits += 1;
        last_hit = Some(hit);
    }

    unique_hits % 2 == 1
}

fn point_on_boundary(prepared: &PreparedBrep, point: Vector3, tolerance: f64) -> bool {
    let tol_sq = tolerance * tolerance;
    prepared
        .triangles
        .iter()
        .any(|triangle| distance_sq(point, closest_point_on_triangle(point, triangle)) <= tol_sq)
}

fn ray_triangle_intersection(
    origin: Vector3,
    direction: Vector3,
    triangle: &TriangleRecord,
    tolerance: f64,
) -> Option<f64> {
    let edge1 = subtract(triangle.b, triangle.a);
    let edge2 = subtract(triangle.c, triangle.a);
    let h = cross(direction, edge2);
    let a = dot(edge1, h);
    if a.abs() <= tolerance {
        return None;
    }

    let f = 1.0 / a;
    let s = subtract(origin, triangle.a);
    let u = f * dot(s, h);
    if u < -tolerance || u > 1.0 + tolerance {
        return None;
    }

    let q = cross(s, edge1);
    let v = f * dot(direction, q);
    if v < -tolerance || u + v > 1.0 + tolerance {
        return None;
    }

    let t = f * dot(edge2, q);
    if t >= -tolerance {
        Some(t)
    } else {
        None
    }
}

fn segment_triangle_intersection(
    start: Vector3,
    end: Vector3,
    triangle: &TriangleRecord,
) -> Option<Vector3> {
    let direction = subtract(end, start);
    let length_sq = dot(direction, direction);
    if length_sq <= 0.0 {
        return None;
    }

    let edge1 = subtract(triangle.b, triangle.a);
    let edge2 = subtract(triangle.c, triangle.a);
    let h = cross(direction, edge2);
    let a = dot(edge1, h);
    if a.abs() <= 1.0e-12 {
        return None;
    }

    let f = 1.0 / a;
    let s = subtract(start, triangle.a);
    let u = f * dot(s, h);
    if !(-1.0e-12..=1.0 + 1.0e-12).contains(&u) {
        return None;
    }

    let q = cross(s, edge1);
    let v = f * dot(direction, q);
    if v < -1.0e-12 || u + v > 1.0 + 1.0e-12 {
        return None;
    }

    let t = f * dot(edge2, q);
    if !(-1.0e-12..=1.0 + 1.0e-12).contains(&t) {
        return None;
    }

    Some(add(start, scale(direction, t)))
}

fn coplanar_triangles_overlap(lhs: &TriangleRecord, rhs: &TriangleRecord) -> bool {
    let (axis_u, axis_v) = dominant_projection_axes(triangle_normal(lhs));
    let lhs_points = [
        project_point(lhs.a, axis_u, axis_v),
        project_point(lhs.b, axis_u, axis_v),
        project_point(lhs.c, axis_u, axis_v),
    ];
    let rhs_points = [
        project_point(rhs.a, axis_u, axis_v),
        project_point(rhs.b, axis_u, axis_v),
        project_point(rhs.c, axis_u, axis_v),
    ];

    if lhs_points
        .iter()
        .any(|point| point_in_triangle_2d(*point, rhs_points[0], rhs_points[1], rhs_points[2]))
    {
        return true;
    }
    if rhs_points
        .iter()
        .any(|point| point_in_triangle_2d(*point, lhs_points[0], lhs_points[1], lhs_points[2]))
    {
        return true;
    }

    let lhs_edges = [
        (lhs_points[0], lhs_points[1]),
        (lhs_points[1], lhs_points[2]),
        (lhs_points[2], lhs_points[0]),
    ];
    let rhs_edges = [
        (rhs_points[0], rhs_points[1]),
        (rhs_points[1], rhs_points[2]),
        (rhs_points[2], rhs_points[0]),
    ];

    for (a0, a1) in lhs_edges {
        for (b0, b1) in rhs_edges {
            if segments_intersect_2d(a0, a1, b0, b1) {
                return true;
            }
        }
    }

    false
}

fn coplanar_overlap_witness(lhs: &TriangleRecord, rhs: &TriangleRecord) -> Vector3 {
    let (axis_u, axis_v) = dominant_projection_axes(triangle_normal(lhs));
    let lhs_points = [
        project_point(lhs.a, axis_u, axis_v),
        project_point(lhs.b, axis_u, axis_v),
        project_point(lhs.c, axis_u, axis_v),
    ];
    let rhs_points = [
        project_point(rhs.a, axis_u, axis_v),
        project_point(rhs.b, axis_u, axis_v),
        project_point(rhs.c, axis_u, axis_v),
    ];

    for point in [lhs.a, lhs.b, lhs.c] {
        let projected = project_point(point, axis_u, axis_v);
        if point_in_triangle_2d(projected, rhs_points[0], rhs_points[1], rhs_points[2]) {
            return point;
        }
    }

    for point in [rhs.a, rhs.b, rhs.c] {
        let projected = project_point(point, axis_u, axis_v);
        if point_in_triangle_2d(projected, lhs_points[0], lhs_points[1], lhs_points[2]) {
            return point;
        }
    }

    lhs.a
}

fn point_in_triangle_2d(p: Point2, a: Point2, b: Point2, c: Point2) -> bool {
    let v0 = subtract_2d(c, a);
    let v1 = subtract_2d(b, a);
    let v2 = subtract_2d(p, a);

    let dot00 = dot_2d(v0, v0);
    let dot01 = dot_2d(v0, v1);
    let dot02 = dot_2d(v0, v2);
    let dot11 = dot_2d(v1, v1);
    let dot12 = dot_2d(v1, v2);

    let denom = dot00 * dot11 - dot01 * dot01;
    if denom.abs() <= 1.0e-24 {
        return false;
    }

    let inv = 1.0 / denom;
    let u = (dot11 * dot02 - dot01 * dot12) * inv;
    let v = (dot00 * dot12 - dot01 * dot02) * inv;
    u >= -1.0e-12 && v >= -1.0e-12 && (u + v) <= 1.0 + 1.0e-12
}

fn segments_intersect_2d(a0: Point2, a1: Point2, b0: Point2, b1: Point2) -> bool {
    let o1 = orientation_2d(a0, a1, b0);
    let o2 = orientation_2d(a0, a1, b1);
    let o3 = orientation_2d(b0, b1, a0);
    let o4 = orientation_2d(b0, b1, a1);

    if o1.abs() <= 1.0e-12 && on_segment_2d(a0, a1, b0) {
        return true;
    }
    if o2.abs() <= 1.0e-12 && on_segment_2d(a0, a1, b1) {
        return true;
    }
    if o3.abs() <= 1.0e-12 && on_segment_2d(b0, b1, a0) {
        return true;
    }
    if o4.abs() <= 1.0e-12 && on_segment_2d(b0, b1, a1) {
        return true;
    }

    (o1 > 0.0 && o2 < 0.0 || o1 < 0.0 && o2 > 0.0) && (o3 > 0.0 && o4 < 0.0 || o3 < 0.0 && o4 > 0.0)
}

fn on_segment_2d(a: Point2, b: Point2, p: Point2) -> bool {
    p.x >= a.x.min(b.x) - 1.0e-12
        && p.x <= a.x.max(b.x) + 1.0e-12
        && p.y >= a.y.min(b.y) - 1.0e-12
        && p.y <= a.y.max(b.y) + 1.0e-12
}

fn orientation_2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn closest_point_on_triangle(point: Vector3, triangle: &TriangleRecord) -> Vector3 {
    let a = triangle.a;
    let b = triangle.b;
    let c = triangle.c;
    let ab = subtract(b, a);
    let ac = subtract(c, a);
    let ap = subtract(point, a);

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = subtract(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }

    let cp = subtract(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let bc = subtract(c, b);
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(bc, w));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add(a, add(scale(ab, v), scale(ac, w)))
}

fn closest_points_on_segments(
    lhs_start: Vector3,
    lhs_end: Vector3,
    rhs_start: Vector3,
    rhs_end: Vector3,
) -> (Vector3, Vector3, f64) {
    let u = subtract(lhs_end, lhs_start);
    let v = subtract(rhs_end, rhs_start);
    let w = subtract(lhs_start, rhs_start);
    let a = dot(u, u);
    let b = dot(u, v);
    let c = dot(v, v);
    let d = dot(u, w);
    let e = dot(v, w);
    let denom = a * c - b * b;
    let mut s_numer;
    let mut s_denom = denom;
    let mut t_numer;
    let mut t_denom = denom;

    if a <= 1.0e-24 && c <= 1.0e-24 {
        return (lhs_start, rhs_start, distance_sq(lhs_start, rhs_start));
    }

    if a <= 1.0e-24 {
        s_numer = 0.0;
        s_denom = 1.0;
        t_numer = e;
        t_denom = c;
    } else if c <= 1.0e-24 {
        t_numer = 0.0;
        t_denom = 1.0;
        s_numer = -d;
        s_denom = a;
    } else if denom <= 1.0e-24 {
        s_numer = 0.0;
        s_denom = 1.0;
        t_numer = e;
        t_denom = c;
    } else {
        s_numer = b * e - c * d;
        t_numer = a * e - b * d;
        if s_numer < 0.0 {
            s_numer = 0.0;
            t_numer = e;
            t_denom = c;
        } else if s_numer > s_denom {
            s_numer = s_denom;
            t_numer = e + b;
            t_denom = c;
        }
    }

    if t_numer < 0.0 {
        t_numer = 0.0;
        if -d < 0.0 {
            s_numer = 0.0;
        } else if -d > a {
            s_numer = s_denom;
        } else {
            s_numer = -d;
            s_denom = a;
        }
    } else if t_numer > t_denom {
        t_numer = t_denom;
        if (-d + b) < 0.0 {
            s_numer = 0.0;
        } else if (-d + b) > a {
            s_numer = s_denom;
        } else {
            s_numer = -d + b;
            s_denom = a;
        }
    }

    let sc = if s_numer.abs() <= 1.0e-24 {
        0.0
    } else {
        s_numer / s_denom
    };
    let tc = if t_numer.abs() <= 1.0e-24 {
        0.0
    } else {
        t_numer / t_denom
    };

    let lhs_point = add(lhs_start, scale(u, sc));
    let rhs_point = add(rhs_start, scale(v, tc));
    (lhs_point, rhs_point, distance_sq(lhs_point, rhs_point))
}

fn triangle_normal(triangle: &TriangleRecord) -> Vector3 {
    cross(
        subtract(triangle.b, triangle.a),
        subtract(triangle.c, triangle.a),
    )
}

fn triangles_are_coplanar(
    lhs: &TriangleRecord,
    rhs: &TriangleRecord,
    lhs_bbox: &BoundingBox,
    rhs_bbox: &BoundingBox,
) -> bool {
    let normal = triangle_normal(lhs);
    let normal_sq = dot(normal, normal);
    if normal_sq <= 1.0e-24 {
        return false;
    }

    let tolerance = lhs_bbox
        .diagonal_length()
        .max(rhs_bbox.diagonal_length())
        .max(1.0)
        * TOUCH_EPSILON_FACTOR;
    let rhs_points = [rhs.a, rhs.b, rhs.c];
    rhs_points
        .iter()
        .all(|point| distance_from_plane(*point, lhs.a, normal) <= tolerance)
}

fn distance_from_plane(point: Vector3, plane_point: Vector3, plane_normal: Vector3) -> f64 {
    let numerator = dot(subtract(point, plane_point), plane_normal).abs();
    let denom = dot(plane_normal, plane_normal).sqrt();
    if denom <= 0.0 {
        f64::INFINITY
    } else {
        numerator / denom
    }
}

fn dominant_projection_axes(normal: Vector3) -> (usize, usize) {
    let ax = normal.x.abs();
    let ay = normal.y.abs();
    let az = normal.z.abs();
    if az >= ax && az >= ay {
        (0, 1)
    } else if ax >= ay {
        (1, 2)
    } else {
        (0, 2)
    }
}

fn project_point(point: Vector3, axis_u: usize, axis_v: usize) -> Point2 {
    Point2 {
        x: component(point, axis_u),
        y: component(point, axis_v),
    }
}

fn component(point: Vector3, axis: usize) -> f64 {
    match axis {
        0 => point.x,
        1 => point.y,
        2 => point.z,
        _ => unreachable!("invalid axis"),
    }
}

fn point_to_json(point: Vector3) -> JsonPoint {
    [point.x, point.y, point.z]
}

fn parse_point_json(point_json: &str) -> Result<Vector3, JsValue> {
    let point = serde_json::from_str::<PointJsonInput>(point_json)
        .map_err(|error| js_error(format!("Invalid point JSON: {error}")))?;

    Ok(match point {
        PointJsonInput::Object { x, y, z } => Vector3::new(x, y, z),
        PointJsonInput::Array([x, y, z]) => Vector3::new(x, y, z),
    })
}

fn has_analytic_geometry(brep: &Brep) -> bool {
    brep.edges
        .iter()
        .any(|edge| matches!(edge.curve, Some(CurveGeometry::Circle { .. })))
        || brep
            .faces
            .iter()
            .any(|face| matches!(face.surface, Some(SurfaceGeometry::Cylinder { .. })))
}

fn accuracy_for_pair(lhs: ProximityAccuracy, rhs: ProximityAccuracy) -> ProximityAccuracy {
    if matches!(lhs, ProximityAccuracy::TessellatedAnalyticSurfaces)
        || matches!(rhs, ProximityAccuracy::TessellatedAnalyticSurfaces)
    {
        ProximityAccuracy::TessellatedAnalyticSurfaces
    } else {
        ProximityAccuracy::ExactPlanarBrep
    }
}

fn ensure_finite_report_pair(report: &BrepProximityReport) -> Result<(), JsValue> {
    if !report.distance.is_finite()
        || !point_is_finite_from_json(report.closest_point_lhs)
        || !point_is_finite_from_json(report.closest_point_rhs)
    {
        return Err(js_error(
            "Proximity result rejected: non-finite pair report",
        ));
    }
    if report.bvh_node_pair_tests == 0 || report.triangle_pair_tests == 0 {
        return Err(js_error(
            "Proximity result rejected: acceleration evidence is incomplete",
        ));
    }
    Ok(())
}

fn ensure_finite_report_point(report: &BrepPointProximityReport) -> Result<(), JsValue> {
    if !report.distance.is_finite()
        || !point_is_finite_from_json(report.query_point)
        || !point_is_finite_from_json(report.closest_point)
    {
        return Err(js_error(
            "Proximity result rejected: non-finite point report",
        ));
    }
    Ok(())
}

fn point_is_finite_from_json(point: JsonPoint) -> bool {
    point[0].is_finite() && point[1].is_finite() && point[2].is_finite()
}

fn point_is_finite(point: Vector3) -> bool {
    point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
}

fn curve_is_finite(curve: &CurveGeometry) -> bool {
    match curve {
        CurveGeometry::Line { start, end } => point_is_finite(*start) && point_is_finite(*end),
        CurveGeometry::Circle {
            center,
            normal,
            x_axis,
            radius,
            start_angle,
            end_angle,
        } => {
            point_is_finite(*center)
                && point_is_finite(*normal)
                && point_is_finite(*x_axis)
                && radius.is_finite()
                && start_angle.is_finite()
                && end_angle.is_finite()
        }
    }
}

fn surface_is_finite(surface: &SurfaceGeometry) -> bool {
    match surface {
        SurfaceGeometry::Plane { origin, normal } => {
            point_is_finite(*origin) && point_is_finite(*normal)
        }
        SurfaceGeometry::Cylinder {
            origin,
            axis,
            ref_direction,
            radius,
            height,
        } => {
            point_is_finite(*origin)
                && point_is_finite(*axis)
                && point_is_finite(*ref_direction)
                && radius.is_finite()
                && height.is_finite()
        }
    }
}

fn triangle_is_finite(a: Vector3, b: Vector3, c: Vector3) -> bool {
    point_is_finite(a) && point_is_finite(b) && point_is_finite(c)
}

fn triangle_area_sq(a: Vector3, b: Vector3, c: Vector3) -> f64 {
    let ab = subtract(b, a);
    let ac = subtract(c, a);
    let normal = cross(ab, ac);
    dot(normal, normal)
}

fn update_best(best: &mut ContactWitness, candidate: ContactWitness) {
    if candidate.distance_sq < best.distance_sq {
        *best = candidate;
    }
}

fn touch_epsilon_sq(lhs_bbox: &BoundingBox, rhs_bbox: &BoundingBox) -> f64 {
    let scale = lhs_bbox
        .diagonal_length()
        .max(rhs_bbox.diagonal_length())
        .max(1.0);
    let tolerance = scale * TOUCH_EPSILON_FACTOR;
    tolerance * tolerance
}

fn distance_sq(a: Vector3, b: Vector3) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

fn dot(a: Vector3, b: Vector3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn add(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

fn subtract(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn scale(a: Vector3, factor: f64) -> Vector3 {
    Vector3::new(a.x * factor, a.y * factor, a.z * factor)
}

#[derive(Clone, Copy)]
struct Point2 {
    x: f64,
    y: f64,
}

fn subtract_2d(a: Point2, b: Point2) -> Point2 {
    Point2 {
        x: a.x - b.x,
        y: a.y - b.y,
    }
}

fn dot_2d(a: Point2, b: Point2) -> f64 {
    a.x * b.x + a.y * b.y
}

impl BoundingBox {
    fn from_points(points: &[Vector3]) -> Option<Self> {
        let first = points.first().copied()?;
        let mut min = first;
        let mut max = first;
        for point in &points[1..] {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            min.z = min.z.min(point.z);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
            max.z = max.z.max(point.z);
        }
        Some(Self { min, max })
    }

    fn center(&self) -> Vector3 {
        Vector3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    fn diagonal_length(&self) -> f64 {
        distance_sq(self.min, self.max).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::builder::BrepBuilder;
    use crate::primitives::cuboid::OGCuboid;
    use crate::primitives::cylinder::OGCylinder;
    use openmaths::Vector3;
    use uuid::Uuid;

    fn parse_pair(json: &str) -> BrepProximityReport {
        serde_json::from_str(json).expect("pair report")
    }

    fn parse_point(json: &str) -> BrepPointProximityReport {
        serde_json::from_str(json).expect("point report")
    }

    fn cuboid(center: Vector3, size: f64) -> Brep {
        let mut primitive = OGCuboid::new("cuboid".to_string());
        primitive
            .set_config(center, size, size, size)
            .expect("cuboid config");
        primitive.world_brep()
    }

    fn cylinder(center: Vector3, radius: f64, height: f64) -> Brep {
        let mut primitive = OGCylinder::new("cylinder".to_string());
        primitive
            .set_config(center, radius, height, 2.0 * std::f64::consts::PI, 24)
            .expect("cylinder config");
        primitive.world_brep()
    }

    fn brep_json(brep: &Brep) -> String {
        serde_json::to_string(brep).unwrap()
    }

    fn point_json(point: Vector3) -> String {
        serde_json::json!([point.x, point.y, point.z]).to_string()
    }

    fn placement_json(position: JsonPoint, rotation: JsonPoint, scale: JsonPoint) -> String {
        serde_json::json!({
            "position": position,
            "rotation": rotation,
            "scale": scale,
        })
        .to_string()
    }

    #[test]
    fn separated_axis_aligned_boxes_return_exact_distance_and_corners() {
        let lhs = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let rhs = cuboid(Vector3::new(4.0, 3.0, 4.0), 2.0);

        let report = parse_pair(
            &calculate_brep_proximity(brep_json(&lhs), brep_json(&rhs)).expect("pair proximity"),
        );

        assert_eq!(report.schema, PAIR_SCHEMA);
        assert_eq!(report.method, METHOD);
        assert_eq!(report.relation, PairRelation::Separated);
        assert_eq!(report.classification, PairClassification::BoundaryDistance);
        assert_eq!(report.source, ProximitySource::BoundaryDistance);
        assert_eq!(report.accuracy, ProximityAccuracy::ExactPlanarBrep);
        assert_eq!(report.lhs_triangle_count, 12);
        assert_eq!(report.rhs_triangle_count, 12);
        assert_eq!(report.acceleration, ProximityAcceleration::TriangleBvhV1);
        assert!(report.bvh_node_pair_tests > 0);
        assert!(report.triangle_pair_tests > 0);
        assert!(report.triangle_pair_tests < 12 * 12);
        assert!((report.distance - 3.0).abs() < 1.0e-9);
        assert!((report.closest_point_lhs[0] - 1.0).abs() < 1.0e-9);
        assert!((report.closest_point_lhs[1] - 1.0).abs() < 1.0e-9);
        assert!((report.closest_point_lhs[2] - 1.0).abs() < 1.0e-9);
        assert!((report.closest_point_rhs[0] - 3.0).abs() < 1.0e-9);
        assert!((report.closest_point_rhs[1] - 2.0).abs() < 1.0e-9);
        assert!((report.closest_point_rhs[2] - 3.0).abs() < 1.0e-9);
    }

    #[test]
    fn touching_boxes_report_zero_clearance_contact() {
        let lhs = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let rhs = cuboid(Vector3::new(2.0, 0.0, 0.0), 2.0);

        let report = parse_pair(
            &calculate_brep_proximity(brep_json(&lhs), brep_json(&rhs)).expect("pair proximity"),
        );

        assert_eq!(report.relation, PairRelation::ZeroClearance);
        assert_eq!(
            report.classification,
            PairClassification::BoundaryContactOrCrossing
        );
        assert_eq!(report.source, ProximitySource::BoundaryContact);
        assert_eq!(report.distance, 0.0);
    }

    #[test]
    fn overlapping_boxes_report_zero_clearance_contact() {
        let lhs = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let rhs = cuboid(Vector3::new(1.0, 0.0, 0.0), 2.0);

        let report = parse_pair(
            &calculate_brep_proximity(brep_json(&lhs), brep_json(&rhs)).expect("pair proximity"),
        );

        assert_eq!(report.relation, PairRelation::ZeroClearance);
        assert_eq!(
            report.classification,
            PairClassification::BoundaryContactOrCrossing
        );
        assert_eq!(report.source, ProximitySource::BoundaryContact);
        assert_eq!(report.distance, 0.0);
    }

    #[test]
    fn fully_contained_boxes_report_zero_clearance_containment() {
        let outer = cuboid(Vector3::new(0.0, 0.0, 0.0), 4.0);
        let inner = cuboid(Vector3::new(0.6, -0.3, 0.2), 1.0);

        let report = parse_pair(
            &calculate_brep_proximity(brep_json(&inner), brep_json(&outer))
                .expect("pair proximity"),
        );

        assert_eq!(report.relation, PairRelation::ZeroClearance);
        assert_eq!(report.classification, PairClassification::Containment);
        assert_eq!(report.source, ProximitySource::Containment);
        assert_eq!(report.distance, 0.0);
        assert_eq!(report.closest_point_lhs, report.closest_point_rhs);
        assert!((report.closest_point_lhs[0] - 0.6).abs() < 1.0e-9);
        assert!((report.closest_point_lhs[1] + 0.3).abs() < 1.0e-9);
        assert!((report.closest_point_lhs[2] - 0.2).abs() < 1.0e-9);
    }

    #[test]
    fn point_outside_returns_nearest_boundary_point() {
        let brep = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let report = parse_point(
            &calculate_brep_point_proximity(
                brep_json(&brep),
                point_json(Vector3::new(3.0, 0.0, 0.0)),
            )
            .expect("point proximity"),
        );

        assert_eq!(report.relation, PointRelation::Outside);
        assert_eq!(report.classification, PointClassification::BoundaryDistance);
        assert_eq!(report.source, ProximitySource::BoundaryDistance);
        assert!((report.distance - 2.0).abs() < 1.0e-9);
        assert!((report.closest_point[0] - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn point_inside_returns_zero_distance_containment() {
        let brep = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let report = parse_point(
            &calculate_brep_point_proximity(
                brep_json(&brep),
                point_json(Vector3::new(0.0, 0.0, 0.0)),
            )
            .expect("point proximity"),
        );

        assert_eq!(report.relation, PointRelation::InsideOrBoundary);
        assert_eq!(report.classification, PointClassification::Containment);
        assert_eq!(report.source, ProximitySource::Containment);
        assert_eq!(report.distance, 0.0);
    }

    #[test]
    fn rigid_transform_preserves_pair_proximity_and_transforms_witnesses() {
        let rotation = Vector3::new(0.0, std::f64::consts::FRAC_PI_2, 0.0);
        let translation = Vector3::new(10.0, -2.0, 4.0);
        let local_lhs = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let local_rhs = cuboid(Vector3::new(4.0, 3.0, 4.0), 2.0);
        let local_report = parse_pair(
            &calculate_brep_proximity(brep_json(&local_lhs), brep_json(&local_rhs))
                .expect("local pair proximity"),
        );

        let mut placement = Placement3D::new();
        placement
            .set_transform(translation, rotation, Vector3::new(1.0, 1.0, 1.0))
            .expect("rigid transform");
        let mut transformed_lhs = local_lhs.clone();
        transformed_lhs.apply_transform(&placement);
        let mut transformed_rhs = local_rhs.clone();
        transformed_rhs.apply_transform(&placement);
        let transformed_report = parse_pair(
            &calculate_brep_proximity(brep_json(&transformed_lhs), brep_json(&transformed_rhs))
                .expect("transformed pair proximity"),
        );

        assert_eq!(transformed_report.relation, local_report.relation);
        assert_eq!(
            transformed_report.classification,
            local_report.classification
        );
        assert_eq!(transformed_report.source, local_report.source);
        assert!((transformed_report.distance - local_report.distance).abs() < 1.0e-9);

        let expected_lhs =
            rotate_and_translate(local_report.closest_point_lhs, rotation, translation);
        let expected_rhs =
            rotate_and_translate(local_report.closest_point_rhs, rotation, translation);
        assert_point_close(transformed_report.closest_point_lhs, expected_lhs);
        assert_point_close(transformed_report.closest_point_rhs, expected_rhs);
    }

    #[test]
    fn cylinder_proximity_marks_tessellated_analytic_surfaces() {
        let lhs = cylinder(Vector3::new(0.0, 0.0, 0.0), 1.0, 2.0);
        let rhs = cylinder(Vector3::new(5.0, 0.0, 0.0), 1.0, 2.0);

        let report = parse_pair(
            &calculate_brep_proximity(brep_json(&lhs), brep_json(&rhs)).expect("pair proximity"),
        );

        assert_eq!(
            report.accuracy,
            ProximityAccuracy::TessellatedAnalyticSurfaces
        );
        assert_eq!(report.relation, PairRelation::Separated);
    }

    #[test]
    fn open_shell_is_rejected() {
        let mut builder = BrepBuilder::new(Uuid::new_v4());
        builder.add_vertices(&[
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ]);
        builder.add_face(&[0, 1, 2], &[]).expect("triangle face");
        let open_shell = builder.build().expect("open shell brep");
        let solid = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);

        assert!(calculate_brep_proximity(brep_json(&open_shell), brep_json(&solid)).is_err());
    }

    #[test]
    fn placement_aware_api_accepts_rotation_and_translation() {
        let lhs = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let rhs = cuboid(Vector3::new(4.0, 3.0, 4.0), 2.0);
        let placement_json = serde_json::json!({
            "position": [10.0, -2.0, 4.0],
            "rotation": [0.0, std::f64::consts::FRAC_PI_2, 0.0],
            "scale": [1.0, 1.0, 1.0],
        })
        .to_string();

        let report = parse_pair(
            &calculate_placed_brep_proximity(
                brep_json(&lhs),
                placement_json.clone(),
                brep_json(&rhs),
                placement_json,
            )
            .expect("placed pair proximity"),
        );

        assert_eq!(report.relation, PairRelation::Separated);
        assert_eq!(report.accuracy, ProximityAccuracy::ExactPlanarBrep);
    }

    #[test]
    fn independent_instance_placements_control_pair_distance() {
        let local = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let identity = placement_json([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let translated = placement_json([4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let report = parse_pair(
            &calculate_placed_brep_proximity(
                brep_json(&local),
                identity,
                brep_json(&local),
                translated,
            )
            .expect("placed pair proximity"),
        );

        assert_eq!(report.relation, PairRelation::Separated);
        assert!((report.distance - 2.0).abs() < 1.0e-9);
        assert!((report.closest_point_lhs[0] - 1.0).abs() < 1.0e-9);
        assert!((report.closest_point_rhs[0] - 3.0).abs() < 1.0e-9);
    }

    #[test]
    fn placed_point_query_uses_world_coordinates() {
        let local = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let translated = placement_json([10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let report = parse_point(
            &calculate_placed_brep_point_proximity(
                brep_json(&local),
                translated,
                point_json(Vector3::new(13.0, 0.0, 0.0)),
            )
            .expect("placed point proximity"),
        );

        assert_eq!(report.relation, PointRelation::Outside);
        assert!((report.distance - 2.0).abs() < 1.0e-9);
        assert!((report.closest_point[0] - 11.0).abs() < 1.0e-9);
    }

    #[test]
    fn non_uniform_placement_is_rejected() {
        let local = cuboid(Vector3::new(0.0, 0.0, 0.0), 2.0);
        let invalid = placement_json([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]);
        assert!(calculate_placed_brep_point_proximity(
            brep_json(&local),
            invalid,
            point_json(Vector3::new(3.0, 0.0, 0.0)),
        )
        .is_err());
    }

    fn assert_point_close(actual: JsonPoint, expected: Vector3) {
        assert!((actual[0] - expected.x).abs() < 1.0e-9, "x mismatch");
        assert!((actual[1] - expected.y).abs() < 1.0e-9, "y mismatch");
        assert!((actual[2] - expected.z).abs() < 1.0e-9, "z mismatch");
    }

    fn rotate_and_translate(point: JsonPoint, rotation: Vector3, translation: Vector3) -> Vector3 {
        let mut transformed = Vector3::new(point[0], point[1], point[2]);
        let mut placement = Placement3D::new();
        placement
            .set_transform(translation, rotation, Vector3::new(1.0, 1.0, 1.0))
            .expect("placement");
        transformed.apply_matrix4(placement.world_matrix());
        transformed
    }
}
