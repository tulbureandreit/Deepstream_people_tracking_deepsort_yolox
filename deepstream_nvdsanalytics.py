import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import pyds

fps_streams={}

MAX_DISPLAY_LEN=64
PGIE_CLASS_ID_VEHICLE=2
PGIE_CLASS_ID_BIKE=1
PGIE_CLASS_ID_PERSON=0

MUXER_OUTPUT_WIDTH=640
MUXER_OUTPUT_HEIGHT=640
MUXER_BATCH_TIMEOUT_USEC=4000000

GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE=0
OSD_DISPLAY_TEXT=1
pgie_classes_str=["person", "bike", "vehicle"]

def nvanalytics_src_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj=frame_meta.obj_meta_list
        obj_counter={
            PGIE_CLASS_ID_BIKE:0,
            PGIE_CLASS_ID_PERSON:0,
            PGIE_CLASS_ID_VEHICLE:0
        }
        while l_obj:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(0.2980, 0.8509, 0.3921, 1.0)
            obj_meta.rect_params.border_width=5
            l_user_meta = obj_meta.obj_user_meta_list

            while l_user_meta:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user_meta)

                    if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                        user_meta_data = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)
                except StopIteration:
                    break

                try:
                    l_user_meta = l_user_meta.next
                except StopIteration:
                    break
            try:
                l_obj=l_obj.next
            except StopIteration:
                break
        
        l_user = frame_meta.frame_user_meta_list
        while l_user:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSFRAME.USER_META"):
                    user_meta_data = pyds.NvDsAnalyticsFrameMeta.cast(user_meta.user_meta_data)
            except StopIteration:
                break
            try:
                l_user=l_user.next
            except StopIteration:
                break

        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        print(fps_streams.values)

        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    if(gstname.find("video") != -1):
        if features.contains("memory:NVMM"):
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to lin decoder src pad to source bin ghost pad")
        else:
            sys.stderr.write("Error: decodebin did not pcik nvidia decoder plugin nvdec_*")
    
def decodebin_child_added(child_proxy, Object, name, user_data):
    if(name.find("decodebin") != -1):
        Object.connect("child-added", decodebin_child_added, user_data)
    
def create_source_bin(index, uri):
    bin_name="source-bin-%02d" %index

    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin")
    
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write("Unable to create uri decode bin")

    uri_decode_bin.set_property("uri", uri)

    uri_decode_bin.connect("pad-added", cb_newpad, nbin)

    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin")
        return None
    return nbin

def main(args):
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN] " %args[0])
        sys.exit(1)
    
    for i in range(0, len(args)-1):
        fps_streams["stream{0}".format(i)]=GETFPS(i)
    number_sources=len(args)-1

    GObject.threads_init()
    Gst.init(None)

    pipeline = Gst.Pipeline()
    is_live = False
    if not pipeline:
        sys.stderr.write("Unable to create pipeline \n")
    
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr("Unable to create NvStreamMux")
    
    pipeline.add(streammux)
    for i in range(number_sources): #in our case this is 1
        uri_name=args[i+1]
        if uri_name.find("rtsp://") == 0:
            is_live=True
        source_bin=create_source_bin(i, uri_name)

        if not source_bin:
            sys.stderr.write("unable to creat source bin")
        
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin")   
        srcpad.link(sinkpad)

        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        if not pgie:
            sys.stderr.write("Unable to create pgie")
        
        tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not tracker:
            sys.stderr.write(" Unable to create tracker")

        nvdsanalytics = Gst.ElementFactory.make("nvdsanalytics", "analytics")
        if not nvdsanalytics:
            sys.stderr.write(" Unable to create nvdsanalytics")

        nvdsanalytics.set_property("config-file", "config_nvdsanalytics.txt")

        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write("Unable to create nvvidconv")

        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if not nvosd:
            sys.stderr.write("Unable to create nvosd")        
        
        nvosd.set_property('process-mode', OSD_PROCESS_MODE)
        nvosd.set_property('display-text', OSD_DISPLAY_TEXT)

        if(is_aarch64()):
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
            if not transform:
                sys.stderr.write("Unable to create nvegltransform")    
            
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write("Unable to create egl sink")   
        if is_live:
            print("One source alive")
            streammux.set_property('live-source', 1)      

        streammux.set_property('width', 640)
        streammux.set_property('height', 640)
        streammux.set_property('batch-size', number_sources) #in our case batch-size is 1, but can have tile of >1 
        streammux.set_property('batched-push-timeout', 4000000)

        pgie.set_property('config-file-path', "YOLOX-deepstream/config_infer_primary.txt")
        pgie_batch_size=pgie.get_property("batch-size")

        if (pgie_batch_size != number_sources): # in our case 1
            pgie.set_property("batch-size", number_sources)
            print("\n WARNING: batch-size is not equal to nr of source. Overriding command!")
        
        sink.set_property('qos', 0)
        sink.set_property('sync', 0) # if we wait to sync tiles, if we have >1 displays, we get less fps

        config = configparser.ConfigParser()
        config.read('dsnvanalytics_tracker_config.txt')
        config.sections()

        for key in config['tracker']:

            if key == 'tracker-width':
                tracker_width = config.getint('tracker', key)
                tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height':
                tracker_height = config.getint('tracker', key)
                tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id':
                tracker_gpu_id = config.getint('tracker', key)
                tracker.set_property('gpu-id', tracker_gpu_id)
            if key == 'll-lib-file':
                tracker_ll_lib_file = config.get('tracker', key)
                tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file':
                tracker_ll_config_file = config.get('tracker', key)
                tracker.set_property('ll-config-file', tracker_ll_config_file)
            if key == 'enable-batch-process': #more helpful when display is tiled  & >1;  in our case are =1
                tracker_enable_batch_process = config.getint('tracker', key)
                tracker.set_property('enable-batch-process', tracker_enable_batch_process)
            if key == 'tracker-past-frames':
                tracker_past_frames = config.getint('tracker', key)
                tracker.set_property('tracker-past-frames', tracker_past_frames)
        
        pipeline.add(pgie)
        pipeline.add(tracker)
        pipeline.add(nvdsanalytics)
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)

        if is_aarch64():
            pipeline.add(transform)
        pipeline.add(sink)

        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(nvdsanalytics)
        nvdsanalytics.link(nvvidconv)
        nvvidconv.link(nvosd)
        if is_aarch64():
            nvosd.link(transform)
            transform.link(sink)

        loop = GObject.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)
        nvdsanalytics_src_pad = nvdsanalytics.get_static_pad("src")
        if not nvdsanalytics_src_pad:
            sys.stderr.write("Unable to create nvdsanalytics_src_pad")  
        else:
            nvdsanalytics_src_pad.add_probe(Gst.PadProbeType.BUFFER, nvanalytics_src_pad_buffer_probe, 0)

        for i, source in enumerate(args):
            if (i !=0):
                print(i, ":", source)
        
        pipeline.set_state(Gst.State.PLAYING)

        try:
            loop.run()
        except:
            pass

        print("\n Exiting NV CARS APP \n")
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))















        






