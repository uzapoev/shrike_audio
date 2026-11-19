cmake_minimum_required(VERSION 3.5)

# Audio module: no GPU, no OS window (depends on mathlib.h from ../, miniaudio (device only) and sse2neon from
# ../../external, pugixml from ../../external/xml and the C++ standard library).
# Standalone:  cmake -S src/audio -B <build dir outside the repo>
# From another project (engine, games):
#   add_subdirectory(<path to src/audio> audio_core)  ->  target_link_libraries(<target> PRIVATE audio_core)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(audio_core CXX)
endif()

set(_ext ${CMAKE_CURRENT_SOURCE_DIR}/../../external)

# pugixml once per build: gui_core compiles pugixml.cpp too - an app linking both must get one copy
if(NOT TARGET pugixml)
    add_library(pugixml STATIC ${_ext}/xml/pugixml.cpp)
    target_include_directories(pugixml PUBLIC ${_ext}/xml)
endif()

add_library(audio_core STATIC
    src/audio.h             src/audio.cpp               # public API + glue
    src/audio_policy.h      src/audio_policy.cpp        # event instances + voice management (game thread)
    src/audio_bank.h        src/audio_bank.cpp          # bank data, XML loader
    src/audio_mixer.h       src/audio_mixer.cpp         # own graph, device (audio thread)
    src/audio_stream.h      src/audio_stream.cpp        # streaming (IO thread)
    src/audio_vorbis.h      src/audio_vorbis.cpp        # Ogg Vorbis decoder (stb_vorbis, patched: external/stb)
    src/audio_plugins.h     src/audio_plugins.cpp       # spatial plugin interface, built-in panning
    src/audio_hrtf.h        src/audio_hrtf.cpp          # built-in binaural plugin (HRTF, hrtf.bin)
    src/audio_geometry.h    src/audio_geometry.cpp      # geometry handles -> plugin
    src/audio_mix.h                                 # SIMD kernels
    src/audio_adpcm.h                               # IMA ADPCM
    src/audio_queue.h                               # SPSC queue
)
target_include_directories(audio_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(audio_core PRIVATE ${_ext})
target_link_libraries(audio_core PUBLIC pugixml)
target_compile_features(audio_core PUBLIC cxx_std_17)
set_target_properties(audio_core PROPERTIES CXX_EXTENSIONS OFF)
if(MSVC)
    target_compile_definitions(audio_core PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
find_package(Threads)
if(Threads_FOUND)
    target_link_libraries(audio_core PUBLIC Threads::Threads)
endif()
if(UNIX AND NOT APPLE AND NOT EMSCRIPTEN)
    target_link_libraries(audio_core PUBLIC dl m)       # miniaudio loads its backends at run time
endif()

# bank tool: source sounds (wav / flac / mp3) -> IMA ADPCM WAV, ready .ogg copied, + sounds.xml. Own miniaudio (decoders),
# not linked to audio_core (compiles audio_vorbis.cpp itself).
option(AUDIO_BUILD_TOOLS "Build audio_bank_tool" ON)
if(AUDIO_BUILD_TOOLS)
    add_executable(audio_bank_tool tools/audio_bank_tool.cpp audio_vorbis.cpp)     # .ogg: checked + decoded like the runtime
    target_include_directories(audio_bank_tool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${_ext})
    target_compile_features(audio_bank_tool PRIVATE cxx_std_17)
    if(MSVC)
        target_compile_definitions(audio_bank_tool PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endif()

# checks (no device; audio_proto_test is the miniaudio prototype and builds on its own)
option(AUDIO_BUILD_TESTS "Build the audio module tests" ON)
if(AUDIO_BUILD_TESTS)
    enable_testing()
    set(_tests ${CMAKE_CURRENT_SOURCE_DIR}/../../tests)
    # kernels: header only; the test has its own miniaudio (to compare with ma_linear_resampler) -> not linked to audio_core
    add_executable(audio_mix_test ${_tests}/audio_mix_test.cpp)
    target_include_directories(audio_mix_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${_ext})
    target_compile_features(audio_mix_test PRIVATE cxx_std_17)
    add_test(NAME audio_mix_test COMMAND audio_mix_test)

    foreach(t audio_mixer_test audio_stream_test audio_policy_test audio_api_test audio_hrtf_test audio_vorbis_test)
        add_executable(${t} ${_tests}/${t}.cpp)
        target_include_directories(${t} PRIVATE ${_tests} ${_ext})
        target_link_libraries(${t} PRIVATE audio_core)
        target_compile_definitions(${t} PRIVATE AUDIO_TEST_DATA="${_tests}/data/audio")     # *.ogg (no encoder in the tests)
        if(MSVC)
            target_compile_definitions(${t} PRIVATE _CRT_SECURE_NO_WARNINGS)
            # tests keep several mixers (~300 KB of queues each) on the stack of one function
            target_link_options(${t} PRIVATE /STACK:8388608)
        endif()
        add_test(NAME ${t} COMMAND ${t})
    endforeach()
endif()
