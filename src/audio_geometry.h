#ifndef __audio_geometry_h__
#define __audio_geometry_h__

// audio_geometry.h - acoustic geometry handles (game thread)
//
// Role:      handles for geometry / geometry instances of the public API, forwarded to the spatial plugin.
// Contents:  audio_geometry_registry: two slot pools (index + generation) mapping handles to plugin ids, the last
//            transform / enabled state of every instance (kept even if the plugin has no geometry).
// Thread:    game thread.
// Not here:  any acoustics - the plugin does it (Steam Audio later); the built-in panning has no geometry, then every
//            call only updates the registry.
// Used by:   audio.cpp. Tests: tests/audio_api_test.cpp.

#include <stdint.h>
#include <vector>

#include "audio_plugins.h"

struct audio_geometry_desc_t;

class audio_geometry_registry
{
public:
    struct handle { uint32_t index = ~0u, generation = 0; };

    void        init(const audio_spatial_plugin& plugin) { m_plugin = plugin; }

    handle      create(const audio_geometry_desc_t& desc);
    void        destroy(handle g);
    handle      create_instance(handle g, const float position[3], const float rotation[4]);
    void        destroy_instance(handle i);
    void        set_transform(handle i, const float position[3], const float rotation[4]);
    void        set_enabled(handle i, bool enabled);

    uint32_t    geometry_count() const;
    uint32_t    instance_count() const;
    bool        instance_enabled(handle i) const;

private:
    struct geometry_slot { uint32_t generation = 0; bool used = false; uint32_t plugin_id = ~0u; uint32_t instances = 0; };
    struct instance_slot { uint32_t generation = 0; bool used = false; uint32_t plugin_id = ~0u; uint32_t geometry = ~0u;
                           float position[3] = { 0, 0, 0 }; float rotation[4] = { 0, 0, 0, 1 }; bool enabled = true; };

    geometry_slot*  geometry(handle g);
    instance_slot*  instance(handle i);

    audio_spatial_plugin        m_plugin = {};
    std::vector<geometry_slot>  m_geometry;
    std::vector<instance_slot>  m_instances;
};

#endif
