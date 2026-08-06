//! Deterministic geometric quantities derived from a validated B-rep solid.
//!
//! The integration runs over each B-rep face's triangulation. Planar faces are
//! therefore integrated from their canonical boundary loops, while analytic
//! curved faces currently use the B-rep's stored tessellation. The report makes
//! that accuracy boundary explicit so callers cannot present a curved result as
//! analytically exact.

use std::fmt;

use openmaths::Vector3;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::operations::triangulate::triangulate_polygon_with_holes;

use super::{validity::check_validity, Brep, SurfaceGeometry};

const VOLUME_RELATIVE_EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MassPropertiesAccuracy {
    ExactPlanarBrep,
    TessellatedAnalyticSurfaces,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AxisAlignedBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BrepMassProperties {
    pub schema: String,
    pub method: String,
    pub accuracy: MassPropertiesAccuracy,
    pub surface_area: f64,
    pub signed_volume: f64,
    pub volume: f64,
    pub centroid: [f64; 3],
    pub bounds: AxisAlignedBounds,
    pub triangle_count: usize,
    pub face_count: usize,
    pub planar_face_count: usize,
    pub analytic_curved_face_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MassPropertiesError {
    InvalidSolid(Vec<String>),
    EmptyGeometry,
    DegenerateVolume,
    NonFiniteResult,
}

impl fmt::Display for MassPropertiesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSolid(issues) => {
                write!(
                    formatter,
                    "BRep is not a valid closed solid: {}",
                    issues.join("; ")
                )
            }
            Self::EmptyGeometry => write!(formatter, "BRep has no triangulatable solid geometry"),
            Self::DegenerateVolume => {
                write!(formatter, "BRep encloses a degenerate or zero volume")
            }
            Self::NonFiniteResult => write!(formatter, "BRep mass properties are non-finite"),
        }
    }
}

/// Wasm entry point for a self-contained, fail-closed quantity calculation.
#[wasm_bindgen(js_name = calculateBrepMassProperties)]
pub fn calculate_brep_mass_properties_wasm(brep_json: String) -> Result<String, JsValue> {
    let brep: Brep = serde_json::from_str(&brep_json)
        .map_err(|error| JsValue::from_str(&format!("Invalid BRep JSON: {error}")))?;
    let properties = calculate_brep_mass_properties(&brep)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serde_json::to_string(&properties).map_err(|error| {
        JsValue::from_str(&format!("Failed to serialize mass properties: {error}"))
    })
}

pub fn calculate_brep_mass_properties(
    brep: &Brep,
) -> Result<BrepMassProperties, MassPropertiesError> {
    let validity = check_validity(brep);
    if !validity.is_valid() {
        return Err(MassPropertiesError::InvalidSolid(validity.issues));
    }

    let bounds = calculate_bounds(brep).ok_or(MassPropertiesError::EmptyGeometry)?;
    let reference = Vector3::new(
        (bounds.min[0] + bounds.max[0]) * 0.5,
        (bounds.min[1] + bounds.max[1]) * 0.5,
        (bounds.min[2] + bounds.max[2]) * 0.5,
    );

    let mut surface_area = 0.0;
    let mut signed_volume = 0.0;
    let mut centroid_numerator = Vector3::new(0.0, 0.0, 0.0);
    let mut triangle_count = 0;

    for face in &brep.faces {
        let (outer, holes) = brep.get_vertices_and_holes_by_face_id(face.id);
        if outer.len() < 3 {
            continue;
        }

        let all_vertices: Vec<Vector3> = outer
            .iter()
            .copied()
            .chain(holes.iter().flatten().copied())
            .collect();
        for triangle in triangulate_polygon_with_holes(&outer, &holes) {
            let a = subtract(all_vertices[triangle[0]], reference);
            let b = subtract(all_vertices[triangle[1]], reference);
            let c = subtract(all_vertices[triangle[2]], reference);
            let cross_bc = cross(b, c);
            let tetrahedron_volume = dot(a, cross_bc) / 6.0;
            let triangle_cross = cross(subtract(b, a), subtract(c, a));

            surface_area += magnitude(triangle_cross) * 0.5;
            signed_volume += tetrahedron_volume;
            centroid_numerator.x += tetrahedron_volume * (a.x + b.x + c.x) * 0.25;
            centroid_numerator.y += tetrahedron_volume * (a.y + b.y + c.y) * 0.25;
            centroid_numerator.z += tetrahedron_volume * (a.z + b.z + c.z) * 0.25;
            triangle_count += 1;
        }
    }

    if triangle_count == 0 {
        return Err(MassPropertiesError::EmptyGeometry);
    }

    let max_extent = (bounds.max[0] - bounds.min[0])
        .max(bounds.max[1] - bounds.min[1])
        .max(bounds.max[2] - bounds.min[2]);
    let volume_epsilon = max_extent.powi(3) * VOLUME_RELATIVE_EPSILON;
    if signed_volume.abs() <= volume_epsilon {
        return Err(MassPropertiesError::DegenerateVolume);
    }

    let centroid = [
        reference.x + centroid_numerator.x / signed_volume,
        reference.y + centroid_numerator.y / signed_volume,
        reference.z + centroid_numerator.z / signed_volume,
    ];
    let volume = signed_volume.abs();
    if !surface_area.is_finite()
        || !signed_volume.is_finite()
        || !volume.is_finite()
        || centroid.iter().any(|coordinate| !coordinate.is_finite())
    {
        return Err(MassPropertiesError::NonFiniteResult);
    }

    let analytic_curved_face_count = brep
        .faces
        .iter()
        .filter(|face| matches!(&face.surface, Some(SurfaceGeometry::Cylinder { .. })))
        .count();
    let planar_face_count = brep.faces.len() - analytic_curved_face_count;
    let accuracy = if analytic_curved_face_count == 0 {
        MassPropertiesAccuracy::ExactPlanarBrep
    } else {
        MassPropertiesAccuracy::TessellatedAnalyticSurfaces
    };

    Ok(BrepMassProperties {
        schema: "opengeometry-brep-mass-properties-v1".to_string(),
        method: "brep-face-triangle-integration".to_string(),
        accuracy,
        surface_area,
        signed_volume,
        volume,
        centroid,
        bounds,
        triangle_count,
        face_count: brep.faces.len(),
        planar_face_count,
        analytic_curved_face_count,
    })
}

fn calculate_bounds(brep: &Brep) -> Option<AxisAlignedBounds> {
    let first = brep.vertices.first()?.position;
    let mut min = [first.x, first.y, first.z];
    let mut max = min;
    for vertex in &brep.vertices[1..] {
        let coordinates = [vertex.position.x, vertex.position.y, vertex.position.z];
        for axis in 0..3 {
            min[axis] = min[axis].min(coordinates[axis]);
            max[axis] = max[axis].max(coordinates[axis]);
        }
    }
    Some(AxisAlignedBounds { min, max })
}

fn subtract(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

fn cross(a: Vector3, b: Vector3) -> Vector3 {
    Vector3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

fn dot(a: Vector3, b: Vector3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn magnitude(vector: Vector3) -> f64 {
    dot(vector, vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::booleans::{boolean_subtraction, types::BooleanOptions};
    use crate::brep::BrepBuilder;
    use crate::primitives::{cuboid::OGCuboid, cylinder::OGCylinder};
    use uuid::Uuid;

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = expected.abs().max(1.0) * 1.0e-9;
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn cuboid_has_exact_volume_area_centroid_and_bounds() {
        let mut cuboid = OGCuboid::new("mass-cuboid".to_string());
        cuboid
            .set_config(Vector3::new(10.0, 20.0, 30.0), 2.0, 3.0, 4.0)
            .expect("cuboid config");

        let properties = calculate_brep_mass_properties(&cuboid.world_brep()).expect("properties");
        assert_eq!(properties.accuracy, MassPropertiesAccuracy::ExactPlanarBrep);
        assert_close(properties.volume, 24.0);
        assert_close(properties.surface_area, 52.0);
        assert_eq!(properties.centroid, [10.0, 20.0, 30.0]);
        assert_eq!(properties.bounds.min, [9.0, 18.5, 28.0]);
        assert_eq!(properties.bounds.max, [11.0, 21.5, 32.0]);
        assert_eq!(properties.face_count, 6);
        assert_eq!(properties.triangle_count, 12);
    }

    #[test]
    fn rigid_placement_and_uniform_scale_transform_quantities() {
        let mut cuboid = OGCuboid::new("placed-mass-cuboid".to_string());
        cuboid
            .set_config(Vector3::new(0.0, 0.0, 0.0), 2.0, 3.0, 4.0)
            .expect("cuboid config");
        cuboid
            .set_transform(
                Vector3::new(5.0, -2.0, 7.0),
                Vector3::new(0.2, -0.4, 0.6),
                Vector3::new(1.5, 1.5, 1.5),
            )
            .expect("placement");

        let properties = calculate_brep_mass_properties(&cuboid.world_brep()).expect("properties");
        assert_close(properties.volume, 24.0 * 1.5_f64.powi(3));
        assert_close(properties.surface_area, 52.0 * 1.5_f64.powi(2));
        assert_close(properties.centroid[0], 5.0);
        assert_close(properties.centroid[1], -2.0);
        assert_close(properties.centroid[2], 7.0);
    }

    #[test]
    fn analytic_curved_surface_is_reported_as_tessellated() {
        let mut cylinder = OGCylinder::new("mass-cylinder".to_string());
        cylinder
            .set_config(
                Vector3::new(0.0, 0.0, 0.0),
                1.0,
                2.0,
                2.0 * std::f64::consts::PI,
                24,
            )
            .expect("cylinder config");

        let properties = calculate_brep_mass_properties(cylinder.brep()).expect("properties");
        assert_eq!(
            properties.accuracy,
            MassPropertiesAccuracy::TessellatedAnalyticSurfaces
        );
        assert!(properties.analytic_curved_face_count > 0);
        assert!(properties.volume < std::f64::consts::PI * 2.0);
    }

    #[test]
    fn wall_opening_volume_uses_boolean_result_instead_of_host_parameters() {
        let mut wall = OGCuboid::new("mass-wall".to_string());
        wall.set_config(Vector3::new(0.0, 1.5, 0.0), 4.0, 3.0, 0.3)
            .expect("wall config");
        let mut opening = OGCuboid::new("mass-opening".to_string());
        opening
            .set_config(Vector3::new(0.0, 1.0, 0.0), 1.0, 2.0, 0.302)
            .expect("opening config");
        let cut = boolean_subtraction(
            &wall.world_brep(),
            &opening.world_brep(),
            BooleanOptions::default(),
        )
        .expect("wall opening subtraction");

        let properties = calculate_brep_mass_properties(&cut.brep).expect("properties");
        assert_eq!(properties.accuracy, MassPropertiesAccuracy::ExactPlanarBrep);
        assert_close(properties.volume, 3.0);
        assert!(properties.volume < 3.6, "opening must reduce host volume");
    }

    #[test]
    fn open_shell_is_rejected() {
        let mut builder = BrepBuilder::new(Uuid::new_v4());
        builder.add_vertices(&[
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ]);
        builder.add_face(&[0, 1, 2], &[]).expect("face");
        let brep = builder.build().expect("brep");

        assert!(matches!(
            calculate_brep_mass_properties(&brep),
            Err(MassPropertiesError::InvalidSolid(_))
        ));
    }
}
