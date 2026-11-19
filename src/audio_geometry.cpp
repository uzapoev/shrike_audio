// audio_geometry.cpp - acoustic geometry handles (summary in audio_geometry.h).
// Parts: slot lookup, geometry create / destroy, instances create / destroy / transform / enable, counters.

#include "audio_geometry.h"

#include <string.h>

audio_geometry_registry::geometry_slot* audio_geometry_registry::geometry(handle g)
{
    if (g.index >= m_geometry.size() || !m_geometry[g.index].used || m_geometry[g.index].generation != g.generation) return nullptr;
    return &m_geometry[g.index];
}

audio_geometry_registry::instance_slot* audio_geometry_registry::instance(handle i)
{
    if (i.index >= m_instances.size() || !m_instances[i.index].used || m_instances[i.index].generation != i.generation) return nullptr;
    return &m_instances[i.index];
}

audio_geometry_registry::handle audio_geometry_registry::create(const audio_geometry_desc_t& desc)
{
    uint32_t idx = 0;
    while (idx < m_geometry.size() && m_geometry[idx].used) ++idx;
    if (idx == m_geometry.size()) m_geometry.push_back(geometry_slot());
    geometry_slot& s = m_geometry[idx];
    s.used = true;
    ++s.generation;
    s.instances = 0;
    s.plugin_id = m_plugin.geometry_create ? m_plugin.geometry_create(m_plugin.user, desc) : ~0u;
    return { idx, s.generation };
}

void audio_geometry_registry::destroy(handle g)
{
    geometry_slot* s = geometry(g);
    if (!s) return;
    // instances of it go first
    for (uint32_t i = 0; i < m_instances.size(); ++i)
        if (m_instances[i].used && m_instances[i].geometry == g.index) destroy_instance({ i, m_instances[i].generation });
    if (m_plugin.geometry_destroy && s->plugin_id != ~0u) m_plugin.geometry_destroy(m_plugin.user, s->plugin_id);
    s->used = false;
}

audio_geometry_registry::handle audio_geometry_registry::create_instance(handle g, const float position[3], const float rotation[4])
{
    geometry_slot* gs = geometry(g);
    if (!gs) return {};
    uint32_t idx = 0;
    while (idx < m_instances.size() && m_instances[idx].used) ++idx;
    if (idx == m_instances.size()) m_instances.push_back(instance_slot());
    instance_slot& s = m_instances[idx];
    s.used = true;
    ++s.generation;
    s.geometry = g.index;
    memcpy(s.position, position, 12);
    memcpy(s.rotation, rotation, 16);
    s.enabled = true;
    s.plugin_id = (m_plugin.geometry_instance_create && gs->plugin_id != ~0u) ? m_plugin.geometry_instance_create(m_plugin.user, gs->plugin_id, position, rotation) : ~0u;
    ++gs->instances;
    return { idx, s.generation };
}

void audio_geometry_registry::destroy_instance(handle i)
{
    instance_slot* s = instance(i);
    if (!s) return;
    if (m_plugin.geometry_instance_destroy && s->plugin_id != ~0u) m_plugin.geometry_instance_destroy(m_plugin.user, s->plugin_id);
    if (s->geometry < m_geometry.size() && m_geometry[s->geometry].instances) --m_geometry[s->geometry].instances;
    s->used = false;
}

void audio_geometry_registry::set_transform(handle i, const float position[3], const float rotation[4])
{
    instance_slot* s = instance(i);
    if (!s) return;
    memcpy(s->position, position, 12);
    memcpy(s->rotation, rotation, 16);
    if (m_plugin.geometry_instance_transform && s->plugin_id != ~0u) m_plugin.geometry_instance_transform(m_plugin.user, s->plugin_id, position, rotation);
}

void audio_geometry_registry::set_enabled(handle i, bool enabled)
{
    instance_slot* s = instance(i);
    if (!s) return;
    s->enabled = enabled;
    if (m_plugin.geometry_instance_enable && s->plugin_id != ~0u) m_plugin.geometry_instance_enable(m_plugin.user, s->plugin_id, enabled);
}

uint32_t audio_geometry_registry::geometry_count() const { uint32_t n = 0; for (auto& s : m_geometry) n += s.used; return n; }
uint32_t audio_geometry_registry::instance_count() const { uint32_t n = 0; for (auto& s : m_instances) n += s.used; return n; }

bool audio_geometry_registry::instance_enabled(handle i) const
{
    if (i.index >= m_instances.size() || !m_instances[i.index].used || m_instances[i.index].generation != i.generation) return false;
    return m_instances[i.index].enabled;
}
