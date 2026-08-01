use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::brep::{Brep, BrepBuilder};
use crate::export::projection::{project_brep_to_scene, CameraParameters, HlrOptions, Scene2D};
use crate::spatial::placement::Placement3D;
use openmaths::Vector3;
use uuid::Uuid;

const MIN_RADIUS: f64 = crate::tolerance::MODELING_TOLERANCE_FLOOR;
const MIN_HEIGHT: f64 = crate::tolerance::MODELING_TOLERANCE_FLOOR;
const MAX_SEGMENTS: u32 = 4096;

#[wasm_bindgen]
#[derive(Clone, Serialize, Deserialize)]
pub struct OGFrustum {
    id: String,
    center: Vector3,
    bottom_radius: f64,
    top_radius: f64,
    height: f64,
    segments: u32,
    start_angle_rad: f64,
    placement: Placement3D,
    brep: Brep,
}

#[wasm_bindgen]
impl OGFrustum {
    #[wasm_bindgen(constructor)]
    pub fn new(id: String) -> OGFrustum {
        OGFrustum {
            id,
            center: Vector3::new(0.0, 0.0, 0.0),
            bottom_radius: 1.0,
            top_radius: 0.0,
            height: 1.0,
            segments: 32,
            start_angle_rad: 0.0,
            placement: Placement3D::new(),
            brep: Brep::new(Uuid::new_v4()),
        }
    }

    #[wasm_bindgen(setter)]
    pub fn set_id(&mut self, id: String) {
        self.id = id;
    }

    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    #[wasm_bindgen]
    pub fn set_config(
        &mut self,
        center: Vector3,
        bottom_radius: f64,
        top_radius: f64,
        height: f64,
        segments: u32,
        start_angle_rad: f64,
    ) -> Result<(), JsValue> {
        validate_frustum_config(center, bottom_radius, top_radius, height, start_angle_rad)
            .map_err(|error| JsValue::from_str(&error))?;

        self.center = center;
        self.bottom_radius = bottom_radius;
        self.top_radius = top_radius;
        self.height = height;
        self.segments = segments.clamp(3, MAX_SEGMENTS);
        self.start_angle_rad = start_angle_rad;
        self.placement.set_anchor(self.center);
        self.generate_brep()
    }

    #[wasm_bindgen]
    pub fn set_transform(
        &mut self,
        position: Vector3,
        rotation: Vector3,
        scale: Vector3,
    ) -> Result<(), JsValue> {
        self.placement
            .set_transform(position, rotation, scale)
            .map_err(|error| JsValue::from_str(&error))
    }

    #[wasm_bindgen]
    pub fn set_translation(&mut self, translation: Vector3) {
        self.placement.set_translation(translation);
    }

    #[wasm_bindgen]
    pub fn set_rotation(&mut self, rotation: Vector3) {
        self.placement.set_rotation(rotation);
    }

    #[wasm_bindgen]
    pub fn set_scale(&mut self, scale: Vector3) -> Result<(), JsValue> {
        self.placement
            .set_scale(scale)
            .map_err(|error| JsValue::from_str(&error))
    }

    pub fn generate_brep(&mut self) -> Result<(), JsValue> {
        self.brep = build_frustum_brep(
            self.brep.id,
            self.bottom_radius,
            self.top_radius,
            self.height,
            self.segments,
            self.start_angle_rad,
        )?;
        self.brep
            .validate_topology()
            .map_err(|error| JsValue::from_str(&format!("Invalid frustum topology: {}", error)))
    }

    #[wasm_bindgen]
    pub fn generate_geometry(&mut self) -> Result<(), JsValue> {
        self.generate_brep()
    }

    #[wasm_bindgen]
    pub fn get_brep_serialized(&self) -> String {
        serde_json::to_string(&self.world_brep()).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_local_brep_serialized(&self) -> String {
        serde_json::to_string(&self.brep).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_geometry_serialized(&self) -> String {
        serde_json::to_string(&self.world_brep().get_triangle_vertex_buffer()).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_local_geometry_serialized(&self) -> String {
        serde_json::to_string(&self.brep.get_triangle_vertex_buffer()).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_geometry_buffer(&self) -> Vec<f64> {
        self.world_brep().get_triangle_vertex_buffer()
    }

    #[wasm_bindgen]
    pub fn get_local_geometry_buffer(&self) -> Vec<f64> {
        self.brep.get_triangle_vertex_buffer()
    }

    #[wasm_bindgen]
    pub fn get_outline_geometry_serialized(&self) -> String {
        serde_json::to_string(&self.world_brep().get_outline_vertex_buffer()).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_local_outline_geometry_serialized(&self) -> String {
        serde_json::to_string(&self.brep.get_outline_vertex_buffer()).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_outline_geometry_buffer(&self) -> Vec<f64> {
        self.world_brep().get_outline_vertex_buffer()
    }

    #[wasm_bindgen]
    pub fn get_local_outline_geometry_buffer(&self) -> Vec<f64> {
        self.brep.get_outline_vertex_buffer()
    }

    #[wasm_bindgen]
    pub fn get_anchor(&self) -> Vector3 {
        self.placement.anchor
    }
}

impl OGFrustum {
    pub fn brep(&self) -> &Brep {
        &self.brep
    }

    pub fn world_brep(&self) -> Brep {
        self.brep.transformed(&self.placement)
    }

    pub fn to_projected_scene2d(&self, camera: &CameraParameters, hlr: &HlrOptions) -> Scene2D {
        project_brep_to_scene(&self.world_brep(), camera, hlr)
    }
}

pub fn validate_frustum_config(
    center: Vector3,
    bottom_radius: f64,
    top_radius: f64,
    height: f64,
    start_angle_rad: f64,
) -> Result<(), String> {
    if !center.x.is_finite() || !center.y.is_finite() || !center.z.is_finite() {
        return Err("Frustum center must be finite.".to_string());
    }
    if !bottom_radius.is_finite() || bottom_radius <= MIN_RADIUS {
        return Err("Frustum bottom radius must be a finite positive value.".to_string());
    }
    if !top_radius.is_finite() || top_radius < 0.0 {
        return Err("Frustum top radius must be a finite non-negative value.".to_string());
    }
    if !height.is_finite() || height <= MIN_HEIGHT {
        return Err("Frustum height must be a finite positive value.".to_string());
    }
    if !start_angle_rad.is_finite() {
        return Err("Frustum start angle must be finite.".to_string());
    }
    Ok(())
}

fn build_ring(radius: f64, y: f64, segments: u32, start_angle_rad: f64) -> Vec<Vector3> {
    (0..segments)
        .map(|index| {
            let angle = start_angle_rad + std::f64::consts::TAU * index as f64 / segments as f64;
            Vector3::new(radius * angle.cos(), y, radius * angle.sin())
        })
        .collect()
}

fn build_frustum_brep(
    id: Uuid,
    bottom_radius: f64,
    top_radius: f64,
    height: f64,
    segments: u32,
    start_angle_rad: f64,
) -> Result<Brep, JsValue> {
    let half_height = height / 2.0;
    let bottom = build_ring(bottom_radius, -half_height, segments, start_angle_rad);
    let mut builder = BrepBuilder::new(id);
    let bottom_ids = builder.add_vertices(&bottom);

    builder.add_face(&bottom_ids, &[]).map_err(|error| {
        JsValue::from_str(&format!("Failed to build frustum bottom face: {}", error))
    })?;

    if top_radius <= MIN_RADIUS {
        let apex_id = builder.add_vertex(Vector3::new(0.0, half_height, 0.0));
        for index in 0..segments as usize {
            let next = (index + 1) % segments as usize;
            builder
                .add_face(&[bottom_ids[next], bottom_ids[index], apex_id], &[])
                .map_err(|error| {
                    JsValue::from_str(&format!(
                        "Failed to build cone side face {}: {}",
                        index, error
                    ))
                })?;
        }
    } else {
        let top = build_ring(top_radius, half_height, segments, start_angle_rad);
        let top_ids = builder.add_vertices(&top);
        let top_face: Vec<u32> = top_ids.iter().rev().copied().collect();
        builder.add_face(&top_face, &[]).map_err(|error| {
            JsValue::from_str(&format!("Failed to build frustum top face: {}", error))
        })?;

        for index in 0..segments as usize {
            let next = (index + 1) % segments as usize;
            builder
                .add_face(
                    &[
                        bottom_ids[next],
                        bottom_ids[index],
                        top_ids[index],
                        top_ids[next],
                    ],
                    &[],
                )
                .map_err(|error| {
                    JsValue::from_str(&format!(
                        "Failed to build frustum side face {}: {}",
                        index, error
                    ))
                })?;
        }
    }

    builder
        .add_shell_from_all_faces(true)
        .map_err(|error| JsValue::from_str(&format!("Failed to build frustum shell: {}", error)))?;
    builder
        .build()
        .map_err(|error| JsValue::from_str(&format!("Failed to finalize frustum BRep: {}", error)))
}
