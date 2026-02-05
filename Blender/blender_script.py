import bpy
from pathlib import Path
import os
import shutil
import bpy.utils.previews
import subprocess
import threading
import json
import time
import uuid

addon_dir = Path(__file__).resolve().parent

brushes = [
    #    {
    #        "name": "Background",
    #        "color": (0, 0, 0),
    #    },
    {
        "name": "Anomaly_0",
        "color": (0 / 255, 158 / 255, 115 / 255)
    },
    {
        "name": "Anomaly_1",
        "color": (213 / 255, 94 / 255, 0 / 255)
    },
    {
        "name": "Anomaly_2",
        "color": (0 / 255, 114 / 255, 178 / 255)
    },
    {
        "name": "Anomaly_3",
        "color": (240 / 255, 228 / 255, 66 / 255)
    },
    {
        "name": "Anomaly_4",
        "color": (204 / 255, 121 / 255, 167 / 255)
    }
]

img_dir = addon_dir / "img_dir"
print("ADDON DIR", img_dir)
img_dir.mkdir(exist_ok=True)
(img_dir / "original").mkdir(exist_ok=True)
(img_dir / "synth").mkdir(exist_ok=True)
(img_dir / "temp").mkdir(exist_ok=True)
(img_dir / "noise").mkdir(exist_ok=True)
(img_dir / "dirs").mkdir(exist_ok=True)
(img_dir / "gn_stats").mkdir(exist_ok=True)

preview_collections = {}
global_state = {}

def get_project_root():
    prefs = bpy.context.preferences.addons[__name__].preferences
    if not prefs.project_root:
        return None
    p = Path(prefs.project_root)
    if not (p / "Blender").exists():
        return None
    return p

def get_object_uuid(crt_object):
    if "persistent_id" not in crt_object:
        crt_object["persistent_id"] = str(uuid.uuid4())
    obj_id = crt_object["persistent_id"]
    return obj_id


def update_texture(synth_fp):
    print("Updating", synth_fp)
    synth_path = Path(synth_fp)
    temp_path = synth_path.parent.parent / "temp" / synth_path.name
    shutil.copy(synth_path, temp_path)

    crt_object = [o for o in bpy.context.scene.objects if hasattr(o, 'select_get') and o.select_get()][0]
    mat = crt_object.active_material
    image = get_image_from_material(mat)
    image.reload()


def process_answer(task_answer):
    print("Latency (s)", (time.time() - task_answer["task"]["start"]))
    if task_answer["result"] == "ERROR":
        print("Error on synthesis; aborting")
    else:
        if task_answer["result"].startswith('Preloaded'):
            print("Done preloading")
        else:
            update_texture(task_answer["result"])


def read_output(process):
    """Thread to read output from the interactive script."""
    for line in process.stdout:
        print(f"[Child]: {line.strip()}")
        if line.startswith("[MSG]"):
            msg = json.loads(line[7:])
            process_answer(msg)
        if "DONE" in line:
            print("[Blender]: Child finished a task.")


def clean_previous_interactive():
    # Clean previous interactive if started
    try:
        global_state["child_process"].terminate()
        global_state["child_process"].stdout.close()
        global_state["child_process"].wait()
        del global_state["child_process"]
    except:
        pass

def start_interactive(project_dir):
    clean_previous_interactive()
    print("Starting interactive")
    python_env = str(project_dir / ".venv/bin/python")
    script_path = str(project_dir / "Blender" / "interactive_synthesis.py")

    child_process = subprocess.Popen(
        [python_env, '-u', script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # line-buffered
    )
    global_state["child_process"] = child_process

    # Start reading output in a separate thread
    thread = threading.Thread(target=read_output, args=(child_process,), daemon=True)
    thread.start()


def send_interactive_task(**kwargs):
    kwargs["start"] = time.time()
    child_process = global_state["child_process"]
    msg = json.dumps(kwargs)
    print(f"[Blender]: Sending command -> {msg}")
    child_process.stdin.write(msg + '\n')
    child_process.stdin.flush()


def load_custom_icons(icons_dir):
    pcoll = bpy.utils.previews.new()

    for i in range(5):
        pcoll.load(f"circle_{i}", os.path.join(icons_dir, f"circle_{i}.png"), 'IMAGE')
    preview_collections["main"] = pcoll


def get_image_from_material(mat):
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image and node.name == "Image Texture":
            image = node.image
    return image


def save_editable_in_temp(crt_object):
    mat = crt_object.active_material
    image = get_image_from_material(mat)
    name = Path(image.filepath_raw).name
    save_path = str((img_dir / "temp" / name).resolve())

    if os.path.exists(save_path):
        os.remove(save_path)
    image.filepath_raw = save_path
    image.save(quality=95)
    return image


def set_brush_common_attributes(context):
    context.tool_settings.image_paint.brush.strength = 1.0
    context.tool_settings.image_paint.brush.curve_preset = 'CONSTANT'
    context.scene.tool_settings.unified_paint_settings.size = 35


def set_brush(context, specs):
    context.tool_settings.image_paint.brush.name = specs["name"]
    set_brush_common_attributes(context)
    context.scene.tool_settings.unified_paint_settings.color = specs["color"]


class BrushSelectOperator(bpy.types.Operator):
    bl_idname = "object.my_select_brush"
    bl_label = "Select brush"

    button_id: bpy.props.IntProperty()

    def execute(self, context):
        crt_object = context.selected_objects[0]
        if "weight_id" not in crt_object:
            self.report({'ERROR'}, "enerate a texture before using the brush")
            return {'CANCELLED'}
        else:
            self.report({'INFO'}, f"Brush {self.button_id} selected!")

        # Trigger weight change
        context.scene.my_settings.my_enum = crt_object["weight_id"]

        bpy.ops.object.mode_set(mode='TEXTURE_PAINT')
        set_brush(context, brushes[self.button_id])
        save_editable_in_temp(crt_object)
        return {'FINISHED'}


def selected_weights_trigger(self, context):
    print("Selected", self.my_enum)
    send_interactive_task(method='preload_weights', weight=self.my_enum)

def get_weights_items(self, context):
    if 'weights_options' in global_state:
        return [(w, w, f"Model {w}") for w in global_state['weights_options']]
    return ["No weights loaded yet"]

class SelectWeightsDropdown(bpy.types.PropertyGroup):
    my_enum: bpy.props.EnumProperty(
        name="Texture Model",
        description="Choose a checkpoint",
        items=get_weights_items,
        update=selected_weights_trigger
    )


class GenerateTexture(bpy.types.Operator):
    """Generate new Texture"""
    bl_idname = "object.generate_texture"
    bl_label = "generate"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        crt_object = context.selected_objects[0]
        self.report({'INFO'}, f"Called generate for {crt_object.name}")

        # 2. Create a new material
        mat = bpy.data.materials.new(name="GeneratedMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes

        # 3. Clear existing nodes to start fresh or just find the BSDF
        bsdf = nodes.get("Principled BSDF")

        # 4. Create the Image Texture node
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.location = (-300, 300)

        # 5. Load and assign the image
        obj_id = get_object_uuid(crt_object)
        image_path = str((img_dir / "temp" / f"{obj_id}.jpg").resolve())
        shutil.copy(Path(global_state['icons_dir'] / "loading.jpg").resolve(), image_path)
        try:
            img = bpy.data.images.load(image_path)
            tex_node.image = img
        except:
            print(f"Could not load image at {image_path}")

        # 6. Link the Texture to the BSDF Base Color
        links = mat.node_tree.links
        links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])

        # 7. Assign the material to the object
        if crt_object.data.materials:
            crt_object.data.materials[0] = mat
        else:
            crt_object.data.materials.append(mat)

        crt_object["weight_id"] = context.scene.my_settings.my_enum
        send_interactive_task(method='generate_image', filepath=image_path, guidance=7.0)

        return {'FINISHED'}


class SynthesizeAnomalies(bpy.types.Operator):
    """Synthesize Anomalies"""
    bl_idname = "object.synthesize_anomalies"
    bl_label = "synthesize"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        crt_object = context.selected_objects[0]
        self.report({'INFO'}, f"Called synthesis for {crt_object.name}")
        image = save_editable_in_temp(crt_object)

        send_interactive_task(method='edit_image', filepath=image.filepath_raw, guidance=7.0)

        return {'FINISHED'}


class ResetObject(bpy.types.Operator):
    """Reset Object"""
    bl_idname = "object.reset"
    bl_label = "reset"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        crt_object = context.selected_objects[0]
        self.report({'INFO'}, f"Resetting texture for object {crt_object.name}")

        image = get_image_from_material(crt_object.active_material)
        temp_path = Path(bpy.path.abspath(image.filepath_raw))
        original_path = temp_path.parent.parent / "original" / temp_path.name
        synth_path = temp_path.parent.parent / "synth" / temp_path.name
        shutil.copy(original_path, temp_path)
        shutil.copy(original_path, synth_path)
        image.reload()
        return {'FINISHED'}


class TextureSynthPanel(bpy.types.Panel):
    bl_idname = "texture_synth_panel"

    bl_category = "FeaturePainting"
    bl_label = "Feature Painting Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        project_dir = get_project_root()
        if project_dir is None:
            box = self.layout.box()
            box.label(text="Standalone project path not set", icon='ERROR')
            box.label(text="Set it in Add-on Preferences")
            return
        if global_state.get('child_process', None) is None or project_dir != global_state['project_dir']:
            on_load(project_dir)
            return

        settings = context.scene.my_settings
        self.layout.prop(settings, "my_enum")
        self.layout.operator(GenerateTexture.bl_idname, text="Generate")

        self.layout.label(text="Use the brush to paint semantic maps")
        row = self.layout.row(align=True)
        row.alignment = 'CENTER'
        icons = preview_collections["main"]
        for i in range(5):
            name = f"circle_{i}"
            op = row.operator(BrushSelectOperator.bl_idname, text="", icon_value=icons[name].icon_id)
            op.button_id = i

        self.layout.operator(SynthesizeAnomalies.bl_idname, text="Edit")
        self.layout.operator(ResetObject.bl_idname, text="Reset")


class MyAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    project_root: bpy.props.StringProperty(
        name="FeaturePainting Project Folder",
        description="Path to the FeaturePainting project root directory",
        subtype='DIR_PATH'
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="FeaturePainting Project Location")
        layout.prop(self, "project_root")
        if self.project_root and not os.path.exists(self.project_root):
            layout.label(text="Path does not exist", icon='ERROR')
        if not (Path(self.project_root) / "Blender").exists():
            layout.label(text="Please select the FeaturePainting root directory", icon='ERROR')

def on_load(project_dir):
    global_state['project_dir'] = project_dir
    global_state['icons_dir'] = project_dir / "Blender" / "icons"
    load_custom_icons(global_state['icons_dir'])
    weights_path = project_dir / "synthesis-runs"
    global_state['weights_options'] = sorted([wp.name for wp in weights_path.iterdir()], reverse=True)
    start_interactive(project_dir)
    for scene in bpy.data.scenes:
        settings = scene.my_settings
        if settings:
            selected_weights_trigger(settings, bpy.context)


bl_info = {
    "name": "Texture Synthesis Add-on",
    "author": "Timotei Ardelean",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "description": "Creates textures and paints features on them",
    "category": "Object",
}

classes = [MyAddonPreferences, GenerateTexture, SynthesizeAnomalies, ResetObject, BrushSelectOperator,
           TextureSynthPanel, SelectWeightsDropdown]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.my_settings = bpy.props.PointerProperty(
        type=SelectWeightsDropdown
    )


def unregister():
    del bpy.types.Scene.my_settings
    print("Unregister called")
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()


if __name__ == "__main__":
    register()
