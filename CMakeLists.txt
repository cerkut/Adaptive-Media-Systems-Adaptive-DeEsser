cmake_minimum_required(VERSION 3.15)

project(TestDeEs VERSION 1.0.0)

# JUCE path 
add_subdirectory("../JUCE" build_juce) 

juce_add_plugin(TestDeEs
    # from 'pluginFormats="buildAU,buildVST3"'
    FORMATS VST3 AU
    
    # for AU:
    BUNDLE_ID "com.stefanos.TestDeEs"   
    PLUGIN_MANUFACTURER_CODE "Stef"
    PLUGIN_CODE "DeEs"                 
    
    # Plugin Settings
    PRODUCT_NAME "TestDeEs" 
    PLUGIN_NAME "TestDeEs"
    
    FORMATS AU VST3 Standalone
    PLUGIN_MANUFACTURER_CODE "Stef"
    PLUGIN_CODE "Ades"
)

target_sources(TestDeEs
    PRIVATE
        PluginProcessor.cpp
        PluginProcessor.h
        PluginEditor.cpp
        PluginEditor.h
)

target_compile_definitions(TestDeEs
    PUBLIC
        # JUCEOPTIONS 
        JUCE_STRICT_REFCOUNTEDPOINTER=1
        JUCE_VST3_CAN_REPLACE_VST2=0
        
        # Standard defaults
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
)

target_link_libraries(TestDeEs
    PRIVATE
        # Modules listed in
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_plugin_client
        juce::juce_audio_processors
        juce::juce_audio_utils
        juce::juce_core
        juce::juce_data_structures
        juce::juce_dsp
        juce::juce_events
        juce::juce_graphics
        juce::juce_gui_basics
        juce::juce_gui_extra
    
    PUBLIC
        # compilation flags
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)