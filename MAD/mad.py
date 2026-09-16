bl_info = {
    "name": "MAD (Microphone Audio Driver)",
    "blender": (4, 2, 0),
    "category": "Animation"
}

import bpy
import sounddevice as sd
import numpy as np
import os
import time
import wave

# Globals
current_volume = 0.0
stream = None
should_run = False

# Recording state
wav_file = None
record_path = ""
record_start_frame = 0
last_strip_refresh = 0.0
RECORD_REFRESH = 0.5  # seconds between live waveform strip refreshes

# Device cache
DEVICE_ITEMS = []
DEVICE_INDEX_MAP = {}

def refresh_microphones():
    global DEVICE_ITEMS, DEVICE_INDEX_MAP

    DEVICE_ITEMS.clear()
    DEVICE_INDEX_MAP.clear()

    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                name = device["name"]

                # Filter out corrupted/invalid names
                if not name or len(name) < 3:
                    continue
                if any(ord(c) < 32 for c in name):
                    continue

                key = str(i)
                DEVICE_ITEMS.append((key, name, ""))
                DEVICE_INDEX_MAP[key] = i

    except Exception as e:
        print("MAD: Failed to refresh microphones:", e)

    # Fallback if nothing valid found
    if not DEVICE_ITEMS:
        DEVICE_ITEMS.append(("0", "No Input Devices Found", ""))

# UI Properties
def get_microphone_items(self, context):
    return DEVICE_ITEMS

class AudioRigSettings(bpy.types.PropertyGroup):
    mic_list: bpy.props.EnumProperty(
        name="Microphone",
        description="Select input device",
        items=get_microphone_items
    )
    object_ref: bpy.props.PointerProperty(
        name="Object",
        type=bpy.types.Object,
        description="Select the target object"
    )
    property_path: bpy.props.StringProperty(
        name="Property Path",
        description="Data path to drive (e.g. 'location.0', 'rotation_euler.2', 'scale[0]')",
        default="location.0"
    )
    bone_name: bpy.props.EnumProperty(
        name="Bone",
        description="Select bone to drive (if Armature)",
        items=lambda self, context: (
            [(b.name, b.name, "") for b in self.object_ref.pose.bones]
            if self.object_ref and self.object_ref.type == 'ARMATURE' and hasattr(self.object_ref, "pose") else []
        )
    )
    volume_scale: bpy.props.FloatProperty(name="Volume to Value Scale", default=1.0)
    update_interval: bpy.props.FloatProperty(name="Update Interval (s)", default=0.05, min=0.001, max=1.0)
    record_to_timeline: bpy.props.BoolProperty(
        name="Record to Timeline",
        description="Record the microphone to a WAV file and show it as a growing waveform strip in the Video Sequencer",
        default=True
    )

# Audio callback
def audio_callback(indata, frames, time, status):
    global current_volume
    volume = np.linalg.norm(indata) / frames
    current_volume = min(volume, 1.0)  # clamp to 1.0 for safety
    if wav_file is not None:
        try:
            wav_file.writeframes((indata[:, 0] * 32767.0).astype('<i2').tobytes())
        except Exception as e:
            print(f"MAD WAV write failed: {e}")

# --- Timeline recording helpers ---
def get_record_dir():
    blend = bpy.path.abspath("//")
    if blend and os.path.isdir(blend):
        return blend
    return "/tmp"

def _strips_collection(se):
    # Blender 5.x renamed sequences -> strips; support both
    return se.strips if hasattr(se, "strips") else se.sequences

def _all_strips(se):
    return se.strips_all if hasattr(se, "strips_all") else se.sequences_all

def _remove_live_strip(scene):
    se = scene.sequence_editor
    if se is None:
        return
    for strip in list(_all_strips(se)):
        if strip.name.startswith("MAD Recording"):
            _strips_collection(se).remove(strip)

def _add_live_strip(scene, muted):
    if scene.sequence_editor is None:
        scene.sequence_editor_create()
    se = scene.sequence_editor
    strip = _strips_collection(se).new_sound("MAD Recording", record_path, channel=1, frame_start=record_start_frame)
    strip.mute = muted
    se.active_strip = strip
    # Ask the Sequencer timeline to redraw so the waveform grows visibly
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'SEQUENCE_EDITOR':
                area.tag_redraw()
    return strip

def start_recording(context, samplerate):
    global wav_file, record_path, record_start_frame, last_strip_refresh
    record_path = os.path.join(get_record_dir(), time.strftime("MAD_%Y%m%d_%H%M%S.wav"))
    record_start_frame = context.scene.frame_current
    wf = wave.open(record_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(int(samplerate))
    wav_file = wf
    last_strip_refresh = time.monotonic()
    # Muted while recording so the mic doesn't feed back through the speakers
    _add_live_strip(context.scene, muted=True)

def refresh_recording_strip():
    global last_strip_refresh
    if wav_file is None:
        return
    now = time.monotonic()
    if now - last_strip_refresh < RECORD_REFRESH:
        return
    last_strip_refresh = now
    scene = bpy.context.scene
    _remove_live_strip(scene)
    _add_live_strip(scene, muted=True)

def stop_recording(context):
    global wav_file
    if wav_file is None:
        return None
    try:
        wav_file.close()
    except Exception:
        pass
    wav_file = None
    scene = context.scene
    _remove_live_strip(scene)
    strip = _add_live_strip(scene, muted=False)
    return strip

# Blender-safe update loop
def update_bone_rotation():
    global should_run
    if not should_run:
        return None

    s = bpy.context.scene.audio_rig_settings
    obj = s.object_ref
    # Update the UI audio level property
    bpy.context.scene["mad_audio_level"] = current_volume

    if s.record_to_timeline:
        refresh_recording_strip()

    if not obj:
        return s.update_interval

    try:
        # If Armature and bone_name is set, drive the bone property
        if obj.type == 'ARMATURE' and s.bone_name:
            bone = obj.pose.bones.get(s.bone_name)
            if bone:
                # Default to driving rotation_euler.x, but allow property_path override
                path = s.property_path.split('.')
                target = bone
                for p in path[:-1]:
                    if '[' in p and ']' in p:
                        arr_name, idx = p[:-1].split('[')
                        target = getattr(target, arr_name)[int(idx)]
                    else:
                        target = getattr(target, p)
                last = path[-1]
                if '[' in last and ']' in last:
                    arr_name, idx = last[:-1].split('[')
                    arr = getattr(target, arr_name)
                    arr[int(idx)] = current_volume * s.volume_scale
                else:
                    setattr(target, last, current_volume * s.volume_scale)
            else:
                print("Bone not found")
        else:
            # Drive the property on the object itself
            path = s.property_path.split('.')
            target = obj
            for p in path[:-1]:
                if '[' in p and ']' in p:
                    arr_name, idx = p[:-1].split('[')
                    target = getattr(target, arr_name)[int(idx)]
                else:
                    target = getattr(target, p)
            last = path[-1]
            if '[' in last and ']' in last:
                arr_name, idx = last[:-1].split('[')
                arr = getattr(target, arr_name)
                arr[int(idx)] = current_volume * s.volume_scale
            else:
                setattr(target, last, current_volume * s.volume_scale)
    except Exception as e:
        print(f"Failed to set property: {e}")

    return s.update_interval  # reschedule

# Operators
class AUDIO_OT_Start(bpy.types.Operator):
    bl_idname = "wm.audio_driver_ui_start"
    bl_label = "Start MAD"
    bl_description = "Start MAD audio driver"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global stream, should_run
        s = context.scene.audio_rig_settings
        should_run = True

        try:
            mic_index = int(s.mic_list.split(":")[0])
            stream = sd.InputStream(device=mic_index, channels=1, callback=audio_callback)
            stream.start()
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start mic stream: {e}")
            return {'CANCELLED'}

        if s.record_to_timeline:
            try:
                start_recording(context, stream.samplerate)
            except Exception as e:
                print(f"MAD timeline recording failed to start: {e}")
                self.report({'WARNING'}, f"Timeline recording failed: {e}")

        bpy.app.timers.register(update_bone_rotation)
        return {'FINISHED'}

class AUDIO_OT_Stop(bpy.types.Operator):
    bl_idname = "wm.audio_driver_ui_stop"
    bl_label = "Stop Audio Driver"
    bl_description = "Stop MAD audio driver"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global stream, should_run
        should_run = False
        if stream:
            stream.stop()
            stream.close()
            stream = None
        strip = stop_recording(context)
        if strip is not None:
            self.report({'INFO'}, f"Recording saved: {record_path}")
        return {'FINISHED'}

# UI Panel
class AUDIO_PT_MicDriverPanel(bpy.types.Panel):
    bl_label = "MAD"
    bl_idname = "AUDIO_PT_mic_driver_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MAD"

    def draw(self, context):
        layout = self.layout
        s = context.scene.audio_rig_settings
        global should_run, current_volume

        layout.prop(s, "mic_list")
        layout.operator("wm.audio_refresh_mics", icon='FILE_REFRESH')
        layout.prop(s, "object_ref")
        if s.object_ref and s.object_ref.type == 'ARMATURE':
            layout.prop(s, "bone_name")
        layout.prop(s, "property_path")
        layout.prop(s, "volume_scale")
        layout.prop(s, "update_interval")
        layout.prop(s, "record_to_timeline")

        row = layout.row()
        row.operator("wm.audio_driver_ui_start", text="Start")
        row.operator("wm.audio_driver_ui_stop", text="Stop")

        # Indicator for active state and audio level
        if should_run:
            layout.label(text="Audio Driver: ACTIVE", icon='PLAY')
            # Draw a progress bar for the current audio level
            layout.prop(
                context.scene, 
                '["mad_audio_level"]', 
                text="Audio Level", 
                slider=True
            )
        else:
            layout.label(text="Audio Driver: Inactive", icon='PAUSE')

class AUDIO_OT_RefreshMics(bpy.types.Operator):
    bl_idname = "wm.audio_refresh_mics"
    bl_label = "Refresh Devices"
    
    def execute(self, context):
        refresh_microphones()
        self.report({'INFO'}, "Microphone list refreshed")
        return {'FINISHED'}

# --- Add a property to hold the audio level for UI updates ---
def update_audio_level(self, context):
    pass  # Dummy update, not used

def ensure_audio_level_property():
    if not hasattr(bpy.types.Scene, "mad_audio_level"):
        bpy.types.Scene.mad_audio_level = bpy.props.FloatProperty(
            name="Audio Level",
            description="Current audio input level",
            default=0.0,
            min=0.0,
            max=1.0,
            update=update_audio_level
        )

# Register
classes = (
    AudioRigSettings,
    AUDIO_OT_Start,
    AUDIO_OT_Stop,
    AUDIO_PT_MicDriverPanel,
)

def register():
    ensure_audio_level_property()
    refresh_microphones()
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.audio_rig_settings = bpy.props.PointerProperty(type=AudioRigSettings)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.audio_rig_settings
    if hasattr(bpy.types.Scene, "mad_audio_level"):
        del bpy.types.Scene.mad_audio_level

if __name__ == "__main__":
    register()
