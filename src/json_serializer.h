#ifndef __json_serializer_h__
#define __json_serializer_h__

// json_serializer.h - reflected structs <-> JSON (sajson reader, own writer)
//
// COPY of the engine's src/json_serializer.h (test repo, 2026-11-20), kept while the audio module lives in its own
// repository. Changes against the engine copy (merge them back or drop this file when the module returns):
//   - no common.h: errors go to json::log_error (a sink set by the caller, else stderr); no guid_t / interned_string
//   - vectors of strings / numbers / bools / enums are read (the engine reads only vectors of reflected structs)
//   - integers are read as numbers (64-bit values survive) and written as int64
//   - json_writer::skip_defaults: a field equal to the default-constructed struct's field is not written
//   - json::pretty(): compact text -> indented text (short objects / arrays on one line), json::parse_error()
//
// Usage:
//   struct point { float x = 0, y = 0; ReflectObject(point, ReflectObjectField(x), ReflectObjectField(y)); };
//   point p = json::from_json_string<point>(text);   std::string t = json::to_json_string(p);

#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <json/sajson.h>

typedef enum field_options {
    field_options_none,
    field_options_asset_ref,
    field_options_read_only,
    field_options_hide_in_inspector,
} field_options;

#define ReflectObject( CLASS, ... )                 public: static auto reflection_properties() { using Type = CLASS; return std::make_tuple(__VA_ARGS__); }
#define ReflectObjectExternal( CLASS, ...)          template <> inline auto reflection_properties<CLASS>() { using Type = CLASS; return std::make_tuple(__VA_ARGS__); }
#define ReflectObjectField(FIELD)                   reflection::make_property(&Type::FIELD, #FIELD)
#define ReflectObjectFieldWithKey(KEY, FIELD)       reflection::make_property(&Type::FIELD, KEY)

template<class T> inline auto                       reflection_properties() {  } // "void": the type has no reflection data

// Enum <-> json string. names[i] is the json name of the value i (the enum is sequential from 0):
//   ReflectEnum(gfx_cull, "none", "back", "front");
struct enum_names_t { const char* const* names = nullptr; size_t count = 0; };
template<class T> inline enum_names_t               reflection_enum_names() { return {}; }
#define ReflectEnum( ENUM, ... )                    template <> inline enum_names_t reflection_enum_names<ENUM>() { static const char* const n[] = { __VA_ARGS__ }; return { n, sizeof(n) / sizeof(n[0]) }; }

namespace reflection
{
    struct base_field_info_t {
        const char* name = "";
        const char* type_name = "";
        const int   type_index = -1;
    };

    template<typename Class, typename T>
    struct field_info_t : base_field_info_t
    {
        T Class::* member = nullptr;
    };

    template<size_t I = 0, typename Tuple, typename Func>
    inline void for_each(const Tuple& tuple, Func&& func) {
        if constexpr (I < std::tuple_size_v<Tuple>) {
            func(I, &std::get<I>(tuple));
            for_each<I + 1>(tuple, std::forward<Func>(func));
        }
    }

    struct type_registry {
        static inline std::atomic<int> g_type_counter { 0 };
        template<class T> static inline int get_type_index() noexcept {
            static int index = g_type_counter.fetch_add(1, std::memory_order_acq_rel);
            return index;
        }
    };

    template <typename T>
    class has_properties
    {
        template <typename C> static std::true_type  test(decltype(&C::reflection_properties)*);
        template <typename C> static std::false_type test(...);
    public:
        static constexpr bool value = decltype(test<T>(nullptr))::value;
    };

    template <typename T, typename std::enable_if_t<has_properties<T>::value>* = nullptr>
    auto get_properties() { return T::reflection_properties(); }

    template <typename T, typename std::enable_if_t<!has_properties<T>::value>* = nullptr>
    auto get_properties() { return reflection_properties<T>(); }

    template<typename Class, typename T>
    auto make_property(T Class::* member, const char* name)
    {
        auto type_name = typeid(T).name();
        const auto type_idx = reflection::type_registry::get_type_index<T>();
        return field_info_t<Class, T> { name, type_name, type_idx, member };
    }

    template<class T> constexpr bool is_reflected() { return !std::is_same_v<decltype(get_properties<T>()), void>; }
}

namespace json
{
    // errors of a read: appended to *g_error when set (one per line), else printed
    inline thread_local std::string* g_error = nullptr;
    template<class... A> inline void log_error(const char* fmt, A... a)
    {
        char buf[512];
        snprintf(buf, sizeof(buf), fmt, a...);
        if (g_error) { if (!g_error->empty()) *g_error += "\n"; *g_error += buf; }
        else fprintf(stderr, "%s\n", buf);
    }

    class json_writer {
    public:
        std::vector<char> buffer;
        bool first_element = true;
        bool skip_defaults = false;

        json_writer(size_t reserve_size = 1024) { buffer.reserve(reserve_size); }

        void start_object() { write_char('{'); first_element = true; }
        void end_object()   { write_char('}'); first_element = false; }
        void start_array()  { write_char('['); first_element = true; }
        void end_array()    { write_char(']'); first_element = false; }
        void write_char(char c) { buffer.push_back(c); }
        void write_str_raw(const char* str, size_t len) { buffer.insert(buffer.end(), str, str + len); }

        void add_key(const char* key) {
            if (!first_element) write_char(',');
            write_char('"');
            write_str_raw(key, strlen(key));
            write_char('"');
            write_char(':');
            first_element = false;
        }

        void write_string_value(const char* value) {
            write_char('"');
            while (*value) {
                const unsigned char c = (unsigned char)*value;
                switch (c) {
                    case '"':   write_str_raw("\\\"", 2); break;
                    case '\\':  write_str_raw("\\\\", 2); break;
                    case '\n':  write_str_raw("\\n", 2);  break;
                    case '\r':  write_str_raw("\\r", 2);  break;
                    case '\t':  write_str_raw("\\t", 2);  break;
                    case '\b':  write_str_raw("\\b", 2);  break;
                    case '\f':  write_str_raw("\\f", 2);  break;
                    default:
                        if (c < 0x20) {
                            char esc[8];
                            int n = snprintf(esc, sizeof(esc), "\\u%04x", (unsigned)c);
                            write_str_raw(esc, (size_t)n);
                        } else {
                            write_char((char)c);
                        }
                }
                value++;
            }
            write_char('"');
        }

        void write_float_value(float value) {
            char local_buf[64];
            auto [ptr, ec] = std::to_chars(local_buf, local_buf + sizeof(local_buf), value);
            if (ec == std::errc()) write_str_raw(local_buf, ptr - local_buf);
            else { int len = snprintf(local_buf, sizeof(local_buf), "%.6f", value); write_str_raw(local_buf, len); }
        }

        void write_int_value(int64_t value) {
            char local_buf[32];
            auto [ptr, ec] = std::to_chars(local_buf, local_buf + sizeof(local_buf), value);
            write_str_raw(local_buf, ptr - local_buf);
        }

        void next_element() {
            if (!first_element) write_char(',');
            first_element = false;
        }
    };

    template<class T> inline void serialize_json(const T& object, json_writer& writer);

    // Context of a read: keys without a field are handed to ctx.unused (no_ctx reports them).
    struct no_ctx {
        template<class T, class Ctx>
        void unused(T&, const sajson::string& k, const sajson::value&, Ctx&) {
            log_error("unknown key '%.*s'", int(k.length()), k.data());
        }
    };

    template<class T, class Ctx>
    inline void deserialize_sajson(T& object, const sajson::value& jvalue, Ctx& ctx);

    namespace detail
    {
        template<class T> struct is_vector                  { static bool const value = false; };
        template<class T> struct is_vector<std::vector<T> > { static bool const value = true;  };

        inline bool key_equals(const sajson::string& key, const char* name)
        {
            const size_t length = strlen(name);
            return key.length() == length && memcmp(key.data(), name, length) == 0;
        }

        template <class T> inline void read_enum(const sajson::value& v, T& value, const char* field)
        {
            if (v.get_type() == sajson::TYPE_INTEGER) { value = (T)v.get_integer_value(); return; }
            if (v.get_type() != sajson::TYPE_STRING) { log_error("'%s' must be a name or a number", field); return; }
            const enum_names_t e = reflection_enum_names<T>();
            const char* str = v.as_cstring();
            for (size_t i = 0; i < e.count; ++i) if (!strcmp(str, e.names[i])) { value = (T)i; return; }
            std::string names;
            for (size_t i = 0; i < e.count; ++i) { if (i) names += " / "; names += e.names[i]; }
            log_error("'%s': unknown value '%s' (%s)", field, str, names.c_str());
        }

        template <class T> inline void write_enum(T value, json_writer& writer)
        {
            const enum_names_t e = reflection_enum_names<T>();
            const size_t i = (size_t)value;
            if (i < e.count) writer.write_string_value(e.names[i]);
            else             writer.write_int_value((int64_t)value);
        }

        inline bool is_number(const sajson::value& v) { return v.get_type() == sajson::TYPE_INTEGER || v.get_type() == sajson::TYPE_DOUBLE; }

        // one value: string, number, bool, enum, vector or a reflected struct
        template<class T, class Ctx> inline void read_value(const sajson::value& value, T& out, const char* name, Ctx& ctx)
        {
            if (value.get_type() == sajson::TYPE_NULL) return;
            if constexpr (std::is_same_v<T, std::string>) {
                if (value.get_type() == sajson::TYPE_STRING) out = value.as_cstring();
                else log_error("'%s' must be a string", name);
            } else if constexpr (std::is_same_v<T, bool>) {
                if (value.get_type() == sajson::TYPE_TRUE || value.get_type() == sajson::TYPE_FALSE) out = value.get_type() == sajson::TYPE_TRUE;
                else if (value.get_type() == sajson::TYPE_INTEGER) out = value.get_integer_value() != 0;
                else log_error("'%s' must be true / false", name);
            } else if constexpr (std::is_enum_v<T>) {
                read_enum(value, out, name);
            } else if constexpr (std::is_floating_point_v<T> || std::is_integral_v<T>) {
                if (is_number(value)) out = (T)value.get_number_value();
                else log_error("'%s' must be a number", name);
            } else if constexpr (is_vector<T>::value) {
                if (value.get_type() != sajson::TYPE_ARRAY) { log_error("'%s' must be an array", name); return; }
                out.resize(value.get_length());
                for (size_t i = 0; i < value.get_length(); ++i) read_value(value.get_array_element(i), out[i], name, ctx);
            } else if constexpr (reflection::is_reflected<T>()) {
                if (value.get_type() != sajson::TYPE_OBJECT) { log_error("'%s' must be an object", name); return; }
                deserialize_sajson(out, value, ctx);
            } else {
                log_error("unknown type '%s'", typeid(T).name());
            }
        }
    }

    template<class T, class Ctx>
    inline void deserialize_sajson(T& object, const sajson::value& jvalue, Ctx& ctx)
    {
        if constexpr (reflection::is_reflected<T>())
        {
            size_t found = 0;
            auto properties = reflection::get_properties<T>();
            reflection::for_each(properties, [&](size_t, auto* arg) {
                sajson::string key(arg->name, strlen(arg->name));
                const size_t at = jvalue.find_object_key(key);
                if (at >= jvalue.get_length()) return;          // no such key: the default stays
                ++found;
                detail::read_value(jvalue.get_object_value(at), object.*arg->member, arg->name, ctx);
            });
            const size_t count = jvalue.get_length();
            if (found < count) {
                for (size_t i = 0; i < count; ++i) {
                    sajson::string key = jvalue.get_object_key(i);
                    bool is_field = false;
                    reflection::for_each(properties, [&](size_t, auto* arg) { if (detail::key_equals(key, arg->name)) is_field = true; });
                    if (!is_field) ctx.unused(object, key, jvalue.get_object_value(i), ctx);
                }
            }
        }
    }

    template<class T>
    inline void deserialize_sajson(T& object, const sajson::value& jvalue)
    {
        no_ctx ctx;
        deserialize_sajson(object, jvalue, ctx);
    }

    namespace detail
    {
        // a struct with `static constexpr bool json_keep_defaults = true;` is written in full even with skip_defaults
        // (small value-like structs: { "x": 0, "y": 1 } reads better than { "y": 1 })
        template<class T, class = void> struct keeps_defaults : std::false_type {};
        template<class T> struct keeps_defaults<T, std::void_t<decltype(T::json_keep_defaults)>> : std::bool_constant<T::json_keep_defaults> {};

        template<class T> inline void write_value(const T& value, json_writer& writer)
        {
            if constexpr (std::is_same_v<T, bool>) {
                if (value) writer.write_str_raw("true", 4); else writer.write_str_raw("false", 5);
            } else if constexpr (std::is_same_v<T, std::string>) {
                writer.write_string_value(value.c_str());
            } else if constexpr (std::is_floating_point_v<T>) {
                writer.write_float_value((float)value);
            } else if constexpr (std::is_enum_v<T>) {
                write_enum(value, writer);
            } else if constexpr (std::is_integral_v<T>) {
                writer.write_int_value((int64_t)value);
            } else if constexpr (is_vector<T>::value) {
                writer.start_array();
                for (const auto& e : value) { writer.next_element(); write_value(e, writer); }
                writer.end_array();
            } else if constexpr (reflection::is_reflected<T>()) {
                serialize_json(value, writer);
            }
        }

        template<class T> inline std::string compact(const T& value)
        {
            json_writer w(64);
            w.skip_defaults = true;
            write_value(value, w);
            return std::string(w.buffer.begin(), w.buffer.end());
        }
    }

    // Writes the reflected fields of `object` as a json object (skip_defaults: fields equal to T{}'s are left out).
    template<class T>
    inline void serialize_json(const T& object, json_writer& writer)
    {
        if constexpr (reflection::is_reflected<T>())
        {
            static const T defaults{};
            bool skip = writer.skip_defaults;
            if constexpr (detail::keeps_defaults<T>::value) skip = false;
            writer.start_object();
            reflection::for_each(reflection::get_properties<T>(), [&](size_t, auto* arg) {
                if (skip && detail::compact(object.*arg->member) == detail::compact(defaults.*arg->member)) return;
                writer.add_key(arg->name);
                detail::write_value(object.*arg->member, writer);
            });
            writer.end_object();
        }
    }

    template<class T>
    inline std::string to_json_string(const T& object, bool skip_defaults = false)
    {
        json_writer writer(1024);
        writer.skip_defaults = skip_defaults;
        serialize_json(object, writer);
        return std::string(writer.buffer.begin(), writer.buffer.end());
    }

    // parse `text` into `object`; false + `error` on a syntax error or (with errors_fail) on any reported error
    template<class T, class Ctx>
    inline bool from_json_string(const std::string& text, T& object, std::string& error, Ctx& ctx)
    {
        std::string copy = text;                                // sajson parses in place
        sajson::document doc = sajson::parse(sajson::dynamic_allocation(), sajson::mutable_string_view(copy.size(), &copy[0]));
        if (!doc.is_valid()) {
            char buf[256];
            snprintf(buf, sizeof(buf), "line %d, column %d: %s", int(doc.get_error_line()), int(doc.get_error_column()), doc.get_error_message_as_cstring());
            error = buf;
            return false;
        }
        if (doc.get_root().get_type() != sajson::TYPE_OBJECT) { error = "not a json object"; return false; }
        std::string* prev = g_error;
        g_error = &error;
        deserialize_sajson(object, doc.get_root(), ctx);
        g_error = prev;
        return error.empty();
    }

    template<class T>
    inline bool from_json_string(const std::string& text, T& object, std::string& error)
    {
        no_ctx ctx;
        return from_json_string(text, object, error, ctx);
    }

    // Compact json -> indented: an object / array goes on one line when it has no object inside and fits `width`
    // (counting the indent); else one member per line. Scalars are copied as they are.
    inline std::string pretty(const std::string& in, int width = 110, int indent = 4)
    {
        struct node { char kind = 0; std::string text; std::vector<std::pair<std::string, node>> items; };     // kind: '{', '[' or 0 (scalar)
        size_t i = 0;
        auto ws = [&]() { while (i < in.size() && (in[i] == ' ' || in[i] == '\n' || in[i] == '\r' || in[i] == '\t')) ++i; };
        auto str = [&]() {                                      // a json string token as it is
            size_t b = i++;
            while (i < in.size() && in[i] != '"') { if (in[i] == '\\') ++i; ++i; }
            ++i;
            return in.substr(b, i - b);
        };
        std::function<node()> value = [&]() -> node {
            ws();
            node n;
            if (i >= in.size()) return n;
            if (in[i] == '{' || in[i] == '[') {
                n.kind = in[i];
                const char close = in[i] == '{' ? '}' : ']';
                ++i;
                ws();
                while (i < in.size() && in[i] != close) {
                    std::string key;
                    if (n.kind == '{') { key = str(); ws(); ++i; }       // key, ':'
                    n.items.push_back({ key, value() });
                    ws();
                    if (i < in.size() && in[i] == ',') { ++i; ws(); }
                }
                ++i;
                return n;
            }
            if (in[i] == '"') { n.text = str(); return n; }
            size_t b = i;
            while (i < in.size() && in[i] != ',' && in[i] != '}' && in[i] != ']' && in[i] != ' ' && in[i] != '\n') ++i;
            n.text = in.substr(b, i - b);
            return n;
        };
        std::function<std::string(const node&)> flat = [&](const node& n) -> std::string {
            if (!n.kind) return n.text;
            if (n.items.empty()) return n.kind == '{' ? "{}" : "[]";
            std::string s(1, n.kind);
            s += ' ';
            for (size_t k = 0; k < n.items.size(); ++k) {
                if (k) s += ", ";
                if (n.kind == '{') s += n.items[k].first + ": ";
                s += flat(n.items[k].second);
            }
            s += ' ';
            s += n.kind == '{' ? '}' : ']';
            return s;
        };
        std::function<bool(const node&)> has_object = [&](const node& n) {
            for (const auto& it : n.items) if (it.second.kind == '{' || has_object(it.second)) return true;
            return false;
        };
        std::string out;
        std::function<void(const node&, int, size_t)> print = [&](const node& n, int depth, size_t prefix) {
            const std::string f = flat(n);
            const bool short_enough = int(depth * indent + prefix + f.size()) <= width;
            if (!n.kind || n.items.empty() || (short_enough && (!has_object(n) || n.kind == '['))) { out += f; return; }
            out += n.kind;
            out += '\n';
            for (size_t k = 0; k < n.items.size(); ++k) {
                out.append(size_t((depth + 1) * indent), ' ');
                std::string key;
                if (n.kind == '{') key = n.items[k].first + ": ";
                out += key;
                print(n.items[k].second, depth + 1, key.size());
                if (k + 1 < n.items.size()) out += ',';
                out += '\n';
            }
            out.append(size_t(depth * indent), ' ');
            out += n.kind == '{' ? '}' : ']';
        };
        node root = value();
        print(root, 0, 0);
        out += '\n';
        return out;
    }
}

#endif
