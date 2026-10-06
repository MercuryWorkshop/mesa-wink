# SPDX-License-Identifier: MIT
extends Node3D

var frame := 0

func _ready():
	print("WINK_FIXTURE godot initialized")
	print("WINK_RENDERER ", RenderingServer.get_video_adapter_name())
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color(0.12, 0.18, 0.25)
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color.WHITE
	environment.environment.ambient_light_energy = 0.25
	add_child(environment)
	var camera := Camera3D.new()
	camera.position = Vector3(0, 3, 7)
	add_child(camera)
	camera.look_at(Vector3(0, 0, 0))
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-55, -30, 0)
	light.shadow_enabled = true
	add_child(light)
	for index in range(6):
		var mesh := MeshInstance3D.new()
		mesh.mesh = SphereMesh.new() if index % 2 == 0 else BoxMesh.new()
		mesh.position = Vector3((index % 3 - 1) * 1.6, 0, (index / 3) * -1.6)
		var material := StandardMaterial3D.new()
		material.albedo_color = Color.from_hsv(index / 6.0, 0.75, 0.8)
		material.metallic = 0.9 if index % 2 else 0.0
		material.roughness = 0.2 + (index % 3) * 0.3
		mesh.material_override = material
		add_child(mesh)
	var floor_mesh := MeshInstance3D.new()
	floor_mesh.mesh = PlaneMesh.new()
	floor_mesh.mesh.size = Vector2(12, 12)
	floor_mesh.position.y = -0.5
	add_child(floor_mesh)
	var overlay := CanvasLayer.new()
	add_child(overlay)
	var label := Label.new()
	label.text = "GL / GLES / Vulkan: text, PBR, depth, shadows"
	label.position = Vector2(12, 12)
	overlay.add_child(label)

func _process(_delta):
	# Frame count, never wall-clock time, controls deterministic state.
	frame += 1
	if frame == 96:
		print("WINK_FIXTURE godot complete")
		get_tree().quit()
