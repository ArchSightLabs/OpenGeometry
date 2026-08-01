use opengeometry::primitives::frustum::{validate_frustum_config, OGFrustum};
use opengeometry::primitives::line::OGLine;
use opengeometry::primitives::rectangle::OGRectangle;
use opengeometry::primitives::sphere::OGSphere;
use openmaths::Vector3;

#[test]
fn sphere_geometry_and_outline_are_non_empty() {
    let mut sphere = OGSphere::new("sphere-smoke".to_string());
    sphere
        .set_config(Vector3::new(0.0, 0.0, 0.0), 1.0, 16, 10)
        .unwrap();

    assert!(!sphere.brep().vertices.is_empty());
    assert!(!sphere.brep().edges.is_empty());
    assert!(!sphere.brep().faces.is_empty());

    let geometry: Vec<f64> = serde_json::from_str(&sphere.get_geometry_serialized()).unwrap();
    let outline: Vec<f64> =
        serde_json::from_str(&sphere.get_outline_geometry_serialized()).unwrap();

    assert!(!geometry.is_empty());
    assert!(!outline.is_empty());
    assert_eq!(geometry.len() % 9, 0);
    assert_eq!(outline.len() % 6, 0);
}

#[test]
fn sphere_segment_inputs_are_clamped() {
    let mut sphere = OGSphere::new("sphere-clamp".to_string());
    sphere
        .set_config(Vector3::new(0.0, 0.0, 0.0), 1.0, 1, 1)
        .unwrap();

    assert!(!sphere.brep().vertices.is_empty());
    assert!(!sphere.brep().faces.is_empty());
}

#[test]
fn line_offset_smoke() {
    let mut line = OGLine::new("line-smoke".to_string());
    line.set_config(Vector3::new(-1.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0))
        .unwrap();
    line.generate_geometry().unwrap();

    let result = line.get_offset_result(0.25, 35.0, true);
    assert_eq!(result.points.len(), 2);
}

#[test]
fn rectangle_generates_face_loop_without_duplicate_halfedges() {
    let mut rectangle = OGRectangle::new("rectangle-smoke".to_string());
    rectangle
        .set_config(Vector3::new(0.0, 0.0, 0.0), 2.0, 1.0)
        .unwrap();
    rectangle.generate_geometry().unwrap();

    assert_eq!(rectangle.brep().faces.len(), 1);
    assert!(rectangle.brep().wires.is_empty());

    let geometry: Vec<f64> = serde_json::from_str(&rectangle.get_geometry_serialized()).unwrap();
    assert_eq!(geometry.len(), 15);
}

#[test]
fn frustum_and_cone_generate_closed_calculation_breps() {
    let mut frustum = OGFrustum::new("frustum-smoke".to_string());
    frustum
        .set_config(Vector3::new(0.0, 1.0, 0.0), 1.0, 0.5, 2.0, 8, 0.0)
        .unwrap();
    frustum.brep().validate_topology().unwrap();
    assert_eq!(frustum.brep().vertices.len(), 16);
    assert_eq!(frustum.brep().faces.len(), 10);
    assert!(frustum.brep().shells.iter().all(|shell| shell.is_closed));
    assert!(frustum.brep().faces[0].normal.y < -0.9);
    assert!(frustum.brep().faces[1].normal.y > 0.9);

    let mut cone = OGFrustum::new("cone-smoke".to_string());
    cone.set_config(Vector3::new(0.0, 1.0, 0.0), 1.0, 0.0, 2.0, 8, 0.0)
        .unwrap();
    cone.brep().validate_topology().unwrap();
    assert_eq!(cone.brep().vertices.len(), 9);
    assert_eq!(cone.brep().faces.len(), 9);
    assert!(cone.brep().shells.iter().all(|shell| shell.is_closed));
    assert!(cone.brep().faces[0].normal.y < -0.9);
    for face in cone.brep().faces.iter().skip(1) {
        let vertices = cone.brep().get_vertices_by_face_id(face.id);
        let centroid_x = vertices.iter().map(|point| point.x).sum::<f64>() / vertices.len() as f64;
        let centroid_z = vertices.iter().map(|point| point.z).sum::<f64>() / vertices.len() as f64;
        assert!(face.normal.x * centroid_x + face.normal.z * centroid_z > 0.0);
    }

    let geometry: Vec<f64> = serde_json::from_str(&cone.get_geometry_serialized()).unwrap();
    assert!(!geometry.is_empty());
    assert_eq!(geometry.len() % 9, 0);
    for triangle in geometry.chunks_exact(9) {
        let ab = Vector3::new(
            triangle[3] - triangle[0],
            triangle[4] - triangle[1],
            triangle[5] - triangle[2],
        );
        let ac = Vector3::new(
            triangle[6] - triangle[0],
            triangle[7] - triangle[1],
            triangle[8] - triangle[2],
        );
        let cross = Vector3::new(
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x,
        );
        assert!(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z > 1.0e-18);
    }
}

#[test]
fn frustum_rejects_degenerate_calculation_inputs() {
    assert!(validate_frustum_config(Vector3::new(0.0, 0.0, 0.0), 0.0, 0.0, 2.0, 0.0).is_err());
    assert!(validate_frustum_config(Vector3::new(0.0, 0.0, 0.0), 1.0, 0.0, 0.0, 0.0).is_err());
}
