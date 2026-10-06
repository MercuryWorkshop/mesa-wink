/* SPDX-License-Identifier: MIT
 * Isolated C ABI bridge to Tint.  Shader semantics are transformed in NIR;
 * this translation step only converts valid SPIR-V to valid WGSL.
 */
#include "gex_wgpu_tint.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/tint/api/tint.h"
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/program/clone_context.h"
#include "src/tint/lang/wgsl/program/program_builder.h"
#include "src/tint/lang/wgsl/reader/reader.h"
#include "src/tint/lang/wgsl/resolver/resolve.h"
#include "src/tint/lang/wgsl/writer/writer.h"

namespace {

using namespace tint;

/* Barriers under control flow WGSL cannot prove uniform.
 *
 * Vulkan and WGSL agree that a barrier has to be reached by the whole
 * workgroup or none of it. They disagree about who checks. Vulkan leaves it
 * to the program, and WGSL proves it statically, with an analysis that has
 * to be conservative. A load from a read-write storage buffer is never
 * uniform to it, even at an address every invocation shares. Nor is any
 * component of a vector that one component of gl_LocalInvocationID went
 * into. EEVEE's shadow passes branch on exactly such a load around their
 * barriers, and Tint refuses the module:
 *
 *   'workgroupBarrier' must only be called from uniform control flow
 *
 * The condition of anything that decides whether a barrier is reached --
 * an if around one, a break or continue out of a loop with one in it --
 * is uniform in every Vulkan program that is valid. So broadcasting it
 * changes nothing that program could observe, and WGSL has the broadcast
 * that its analysis accepts: every invocation stores the condition to a
 * workgroup variable, and the branch reads it back with
 * workgroupUniformLoad. That load is two barriers, paid only by modules
 * that were refused without it.
 *
 * What it cannot fix is a return before a barrier. That makes the
 * invocations that took it leave for good, so a broadcast would make the
 * others leave too. A module like that stays refused. */
class UniformBarriers {
public:
   explicit UniformBarriers(const Program &src) : src_(src) {}

   /* The module rewritten, or an invalid Program if the rewrite did not
    * satisfy the analysis. */
   Program Run()
   {
      find_barrier_functions();
      for (const ast::Function *fn : src_.AST().Functions())
         walk(fn->body);

      ProgramBuilder b;
      program::CloneContext ctx(&b, &src_, true);
      for (const ast::Enable *enable : src_.AST().Enables()) {
         if (enable->HasExtension(wgsl::Extension::kChromiumDisableUniformityAnalysis) &&
             enable->extensions.Length() == 1u)
            ctx.Remove(src_.AST().GlobalDeclarations(), enable);
      }
      const Symbol var = b.Symbols().New("wink_uniform_condition");
      b.GlobalVar(var, b.ty.bool_(), core::AddressSpace::kWorkgroup);
      auto broadcast = [&b, var] {
         return b.Call("workgroupUniformLoad", b.AddressOf(var));
      };
      for (const ast::IfStatement *i : ifs_) {
         /* A block in its place, so an else-if can be one too. */
         ctx.Replace(i, [&ctx, &b, var, broadcast, i] {
            const ast::Expression *condition = ctx.Clone(i->condition);
            const ast::BlockStatement *body = ctx.Clone(i->body);
            const ast::Statement *otherwise = ctx.Clone(i->else_statement);
            return b.Block(b.Assign(var, condition),
                           b.If(broadcast(), body, b.Else(otherwise), ctx.Clone(i->attributes)));
         });
      }
      for (const auto &[continuing, break_if] : break_ifs_) {
         const ast::BreakIfStatement *bi = break_if;
         ctx.InsertBefore(continuing->statements, bi, [&ctx, &b, var, bi] {
            return b.Assign(var, ctx.Clone(bi->condition));
         });
         ctx.Replace(bi->condition, broadcast);
      }
      ctx.Clone();
      return resolver::Resolve(b, wgsl::AllowedFeatures::Everything());
   }

private:
   enum class Kind { Loop, Switch, If };
   struct Frame {
      Kind kind;
      const ast::Statement *node;
      bool barrier;
   };

   bool is_barrier(const ast::CallExpression *call) const
   {
      const std::string name = call->target->identifier->symbol.Name();
      return name == "workgroupBarrier" || name == "storageBarrier" ||
             name == "textureBarrier" || name == "workgroupUniformLoad" ||
             barrier_functions_.count(name);
   }

   /* Whether a barrier is anywhere inside `s`. */
   bool has_barrier(const ast::Statement *s)
   {
      if (!s)
         return false;
      if (auto it = memo_.find(s); it != memo_.end())
         return it->second;
      bool found = false;
      if (auto *block = s->As<ast::BlockStatement>()) {
         for (const ast::Statement *c : block->statements)
            found = has_barrier(c) || found;
      } else if (auto *i = s->As<ast::IfStatement>()) {
         found = has_barrier(i->body) || has_barrier(i->else_statement);
      } else if (auto *l = s->As<ast::LoopStatement>()) {
         found = has_barrier(l->body) || has_barrier(l->continuing);
      } else if (auto *f = s->As<ast::ForLoopStatement>()) {
         found = has_barrier(f->initializer) || has_barrier(f->body) || has_barrier(f->continuing);
      } else if (auto *w = s->As<ast::WhileStatement>()) {
         found = has_barrier(w->body);
      } else if (auto *sw = s->As<ast::SwitchStatement>()) {
         for (const ast::CaseStatement *c : sw->body)
            found = has_barrier(c->body) || found;
      } else if (auto *call = s->As<ast::CallStatement>()) {
         found = is_barrier(call->expr);
      }
      memo_[s] = found;
      return found;
   }

   /* A function that reaches a barrier makes its call sites barriers. */
   void find_barrier_functions()
   {
      for (bool changed = true; changed;) {
         changed = false;
         for (const ast::Function *fn : src_.AST().Functions()) {
            const std::string name = fn->name->symbol.Name();
            memo_.clear();
            if (!barrier_functions_.count(name) && has_barrier(fn->body)) {
               barrier_functions_.insert(name);
               changed = true;
            }
         }
      }
      memo_.clear();
   }

   /* The ifs between a break or continue and the loop it leaves decide
    * whether that loop's barrier is reached again. */
   void mark_exit(bool is_break)
   {
      for (size_t n = stack_.size(); n-- > 0;) {
         const Frame &f = stack_[n];
         if (f.kind == Kind::Switch && is_break)
            return;
         if (f.kind == Kind::Loop) {
            if (f.barrier) {
               for (size_t m = n + 1; m < stack_.size(); ++m) {
                  if (stack_[m].kind == Kind::If)
                     ifs_.insert(stack_[m].node->As<ast::IfStatement>());
               }
            }
            return;
         }
      }
   }

   void walk(const ast::Statement *s, const ast::BlockStatement *continuing = nullptr)
   {
      if (!s)
         return;
      if (auto *block = s->As<ast::BlockStatement>()) {
         for (const ast::Statement *c : block->statements)
            walk(c, continuing == block ? block : nullptr);
      } else if (auto *i = s->As<ast::IfStatement>()) {
         if (has_barrier(i))
            ifs_.insert(i);
         stack_.push_back({Kind::If, i, false});
         walk(i->body);
         walk(i->else_statement);
         stack_.pop_back();
      } else if (auto *l = s->As<ast::LoopStatement>()) {
         stack_.push_back({Kind::Loop, l, has_barrier(l)});
         walk(l->body);
         walk(l->continuing, l->continuing);
         stack_.pop_back();
      } else if (auto *f = s->As<ast::ForLoopStatement>()) {
         walk(f->initializer);
         stack_.push_back({Kind::Loop, f, has_barrier(f)});
         walk(f->body);
         walk(f->continuing);
         stack_.pop_back();
      } else if (auto *w = s->As<ast::WhileStatement>()) {
         stack_.push_back({Kind::Loop, w, has_barrier(w)});
         walk(w->body);
         stack_.pop_back();
      } else if (auto *sw = s->As<ast::SwitchStatement>()) {
         stack_.push_back({Kind::Switch, sw, false});
         for (const ast::CaseStatement *c : sw->body)
            walk(c->body);
         stack_.pop_back();
      } else if (s->Is<ast::BreakStatement>()) {
         mark_exit(true);
      } else if (s->Is<ast::ContinueStatement>()) {
         mark_exit(false);
      } else if (auto *bi = s->As<ast::BreakIfStatement>()) {
         if (continuing && !stack_.empty() && stack_.back().kind == Kind::Loop &&
             stack_.back().barrier)
            break_ifs_.push_back({continuing, bi});
      }
   }

   const Program &src_;
   std::unordered_map<const ast::Statement *, bool> memo_;
   std::unordered_set<std::string> barrier_functions_;
   std::vector<Frame> stack_;
   std::unordered_set<const ast::IfStatement *> ifs_;
   std::vector<std::pair<const ast::BlockStatement *, const ast::BreakIfStatement *>> break_ifs_;
};

/* A fragment shader's colour attachments read as they are, which SPIR-V
 * has no way of saying that Tint reads: the inputs named wink_color_N, at
 * whatever location, made @color(N) (chromium_experimental_framebuffer_fetch),
 * without the interpolation that is a location's. */
bool
color_input(const std::string &name, uint32_t *index)
{
   static const char prefix[] = "wink_color_";
   const size_t at = name.find(prefix);
   if (at == std::string::npos || at + sizeof(prefix) - 1 >= name.size())
      return false;
   const char digit = name[at + sizeof(prefix) - 1];
   if (digit < '0' || digit > '7')
      return false;
   *index = digit - '0';
   return true;
}

/* An invalid Program where nothing needed it. */
Program
fetch_colors(const Program &src)
{
   ProgramBuilder b;
   program::CloneContext ctx(&b, &src, true);
   bool any = false;
   auto rewrite = [&](const std::string &name, const auto &attributes) {
      uint32_t index;
      if (!color_input(name, &index))
         return;
      for (const ast::Attribute *attr : attributes) {
         if (attr->Is<ast::LocationAttribute>()) {
            ctx.Replace(attr, b.Color(core::u32(index)));
            any = true;
         } else if (attr->Is<ast::InterpolateAttribute>()) {
            ctx.Remove(attributes, attr);
         }
      }
   };
   for (const ast::Function *fn : src.AST().Functions()) {
      if (!fn->IsEntryPoint() || fn->PipelineStage() != ast::PipelineStage::kFragment)
         continue;
      for (const ast::Parameter *param : fn->params)
         rewrite(param->name->symbol.Name(), param->attributes);
   }
   for (const ast::TypeDecl *decl : src.AST().TypeDecls()) {
      if (const ast::Struct *s = decl->As<ast::Struct>()) {
         for (const ast::StructMember *member : s->members)
            rewrite(member->name->symbol.Name(), member->attributes);
      }
   }
   if (!any)
      return Program{};
   b.Enable(wgsl::Extension::kChromiumExperimentalFramebufferFetch);
   ctx.Clone();
   return resolver::Resolve(b, wgsl::AllowedFeatures::Everything());
}

/* Whether the SPIR-V reader refused a builtin it has none of. */
bool
unknown_builtin(const diag::List &diagnostics)
{
   for (const diag::Diagnostic &d : diagnostics) {
      if (d.severity == diag::Severity::Error &&
          d.message.Plain().find("unknown SPIR-V builtin") != std::string::npos)
         return true;
   }
   return false;
}

/* Whether every error is the uniformity analysis refusing a barrier. */
bool
only_barrier_uniformity(const diag::List &diagnostics)
{
   bool any = false;
   for (const diag::Diagnostic &d : diagnostics) {
      if (d.severity != diag::Severity::Error)
         continue;
      if (d.message.Plain().find("must only be called from uniform control flow") ==
          std::string::npos)
         return false;
      any = true;
   }
   return any;
}

/* The SPIR-V reader's module, printed despite the analysis, read again
 * without it, and rewritten so it holds. */
Program
make_barriers_uniform(const Program &refused, std::string *error)
{
   auto printed = wgsl::writer::Generate(refused, {});
   if (printed != Success) {
      *error = printed.Failure().reason;
      return Program{};
   }
   Source::File file("wink", "enable chromium_disable_uniformity_analysis;\n" +
                                printed.Get().wgsl);
   wgsl::reader::Options options;
   options.allowed_features = wgsl::AllowedFeatures::Everything();
   Program reread = wgsl::reader::Parse(&file, options);
   if (!reread.IsValid()) {
      *error = reread.Diagnostics().Str();
      return Program{};
   }
   Program rewritten = UniformBarriers(reread).Run();
   if (!rewritten.IsValid())
      *error = rewritten.Diagnostics().Str();
   return rewritten;
}

char *copy_string(const std::string& value)
{
   char *copy = static_cast<char *>(std::malloc(value.size() + 1));
   if (!copy)
      return nullptr;
   std::memcpy(copy, value.data(), value.size());
   copy[value.size()] = '\0';
   return copy;
}

void set_error(char **error_out, const std::string& message)
{
   if (error_out)
      *error_out = copy_string(message);
}

} // namespace

static char *
translate(const uint32_t *words, size_t word_count, char **error_out)
{
   if (error_out)
      *error_out = nullptr;
   if (!words || !word_count) {
      set_error(error_out, "empty SPIR-V module");
      return nullptr;
   }

   static std::once_flag initialized;
   std::call_once(initialized, [] { tint::Initialize(); });

   std::vector<uint32_t> spirv(words, words + word_count);
   /* GEX_WGPU_SPIRV_DIR=<dir> writes every module there as N.spv before Tint
    * reads it: for the ones Tint does not come back from. */
   if (const char *dir = getenv("GEX_WGPU_SPIRV_DIR")) {
      static int next = 0;
      char path[1024];
      snprintf(path, sizeof(path), "%s/%d.spv", dir, next++);
      if (FILE *f = fopen(path, "wb")) {
         fwrite(words, 4, word_count, f);
         fclose(f);
      }
   }
   tint::spirv::reader::Options options;
   options.allow_non_uniform_derivatives = true;
   options.allowed_features = tint::wgsl::AllowedFeatures::Everything();
   tint::Program program = tint::spirv::reader::Read(spirv, options);
   /* A builtin the AST reader has none of -- PrimitiveId, which is WGSL's
    * primitive_index -- and the IR reader has: its module, as a program. */
   if (!program.IsValid() && unknown_builtin(program.Diagnostics())) {
      auto ir = tint::spirv::reader::ReadIR(spirv, options);
      if (ir == tint::Success) {
         tint::wgsl::writer::Options writer_options;
         writer_options.allow_non_uniform_derivatives = true;
         writer_options.allowed_features = tint::wgsl::AllowedFeatures::Everything();
         auto from_ir = tint::wgsl::writer::ProgramFromIR(ir.Get(), writer_options);
         if (from_ir == tint::Success)
            program = std::move(from_ir.Get());
         else if (getenv("GEX_WGPU_UNIFORM_LOG"))
            fprintf(stderr, "gex-wgpu: the IR reader's module not a program: %s\n", from_ir.Failure().reason.c_str());
      } else if (getenv("GEX_WGPU_UNIFORM_LOG")) {
         fprintf(stderr, "gex-wgpu: the IR reader refused it too: %s\n", ir.Failure().reason.c_str());
      }
   }
   if (!program.IsValid() && only_barrier_uniformity(program.Diagnostics())) {
      if (getenv("GEX_WGPU_UNIFORM_LOG"))
         fprintf(stderr, "gex-wgpu: barriers made uniform, for: %s\n", program.Diagnostics().Str().c_str());
      std::string why;
      tint::Program rewritten = make_barriers_uniform(program, &why);
      if (rewritten.IsValid())
         program = std::move(rewritten);
      else if (getenv("GEX_WGPU_UNIFORM_LOG"))
         fprintf(stderr, "gex-wgpu: barriers not made uniform: %s\n", why.c_str());
   }
   if (!program.IsValid()) {
      /* The module Tint would not read, as hex, so it can be disassembled.
       * Its complaints name a construct without saying which instruction, and
       * the SPIR-V is the only place the answer is: pipe this through
       * `xxd -r -p > m.spv && spirv-dis m.spv`.
       *
       * Once per process, and only for a module small enough to be worth
       * printing. Not behind an environment variable because this code runs
       * in the host module, whose environment nothing plumbs -- and because
       * reaching here at all means a shader the application asked for does
       * not work, which is already being reported loudly. */
      static bool dumped = false;
      if (!dumped && word_count < 20000) {
         dumped = true;
         fprintf(stderr, "gex-wgpu: rejected SPIR-V (%zu words) hex follows\n",
                 word_count);
         std::string hex;
         hex.reserve(word_count * 8 + 1);
         static const char digits[] = "0123456789abcdef";
         for (size_t i = 0; i < word_count; ++i) {
            /* Little-endian bytes, which is how a .spv file is written. */
            const uint32_t w = words[i];
            for (int b = 0; b < 4; ++b) {
               const unsigned byte = (w >> (8 * b)) & 0xffu;
               hex.push_back(digits[byte >> 4]);
               hex.push_back(digits[byte & 0xf]);
            }
         }
         for (size_t at = 0; at < hex.size(); at += 160)
            fprintf(stderr, "SPIRVHEX %s\n", hex.substr(at, 160).c_str());
         fprintf(stderr, "gex-wgpu: rejected SPIR-V hex ends\n");
      }
      set_error(error_out, "Tint rejected SPIR-V: " + program.Diagnostics().Str());
      return nullptr;
   }

   {
      tint::Program fetched = fetch_colors(program);
      if (fetched.IsValid())
         program = std::move(fetched);
      else if (!fetched.Diagnostics().IsEmpty())
         fprintf(stderr, "gex-wgpu: framebuffer fetch not made: %s\n", fetched.Diagnostics().Str().c_str());
   }
   auto generated = tint::wgsl::writer::Generate(program, {});
   if (generated != tint::Success) {
      set_error(error_out, "Tint WGSL generation failed: " +
                           generated.Failure().reason);
      return nullptr;
   }
   /* GEX_WGPU_DUMP_WGSL=<n> prints the first n modules Tint produces.
    *
    * The missing instrument for every shader that compiles and then computes
    * the wrong thing: the SPIR-V is dumped only when Tint *rejects* it, so a
    * module that translates cleanly and behaves wrongly left nothing to read.
    * Counted rather than gated on a single shader, because the interesting
    * one is rarely the first. */
   {
      static int budget = -1;
      if (budget < 0) {
         const char *want = getenv("GEX_WGPU_DUMP_WGSL");
         budget = want ? atoi(want) : 0;
      }
      if (budget > 0) {
         --budget;
         fprintf(stderr, "==== gex WGSL (%zu bytes) ====\n%s==== end ====\n",
                 generated.Get().wgsl.size(), generated.Get().wgsl.c_str());
      }
   }
   char *wgsl = copy_string(generated.Get().wgsl);
   if (!wgsl)
      set_error(error_out, "out of memory copying WGSL");
   return wgsl;
}

/* The translation cache: SPIR-V in, the WGSL Tint made of it out.
 *
 * Tint is most of a shader compile here (a 600-word module took 341 ms of
 * 351 in a software run; Noita's first menus and its first world frame each
 * wait ~3 s on it), and its answer depends on nothing but the words and this
 * build. So the page keeps the answers: it hands a blob of them in before the
 * first compile (`gex_wgsl_cache_load`, from the site's bundle or the
 * browser's own store) and takes back what was translated since
 * (`gex_wgsl_cache_dump`) to keep for the next load. A blob made by another
 * build is refused whole, so a rebuilt Tint never serves an old answer.
 *
 * Keyed by two independent 64-bit hashes of the words and their count.
 *
 * Held to a budget of bytes of WGSL (GEX_WGSL_CACHE_BYTES, or the page's
 * gex_wgsl_cache_budget; 32 MiB): past it, what was used least recently
 * goes first -- which, after a load, is what the load brought and nothing
 * has asked for. In a browser all of it is wasm heap, whose peak stays. */
namespace {

struct cache_key {
   uint64_t a, b;
   bool operator==(const cache_key &o) const { return a == o.a && b == o.b; }
};
struct cache_key_hash {
   size_t operator()(const cache_key &k) const { return (size_t)(k.a ^ (k.b * 0x9e3779b97f4a7c15ull)); }
};
struct cache_entry {
   std::string wgsl;
   bool fresh; /* translated in this process, not loaded */
   std::list<cache_key>::iterator lru;
};

std::mutex cache_mutex;
std::unordered_map<cache_key, cache_entry, cache_key_hash> cache;
/* Most recently used first. */
std::list<cache_key> cache_lru;
size_t cache_bytes;
size_t cache_budget = 32u << 20;
bool cache_budget_read;
uint32_t cache_hits, cache_misses, cache_loaded, cache_evicted;

/* Under cache_mutex. */
void
read_budget()
{
   if (cache_budget_read)
      return;
   cache_budget_read = true;
   const char *v = getenv("GEX_WGSL_CACHE_BYTES");
   if (v && *v)
      cache_budget = (size_t)strtoull(v, nullptr, 0);
}

void
evict_to(size_t budget)
{
   while (cache_bytes > budget && !cache_lru.empty()) {
      auto it = cache.find(cache_lru.back());
      cache_lru.pop_back();
      if (it == cache.end())
         continue;
      cache_bytes -= it->second.wgsl.size();
      cache.erase(it);
      ++cache_evicted;
   }
}

/* An entry, at the front, or not if its key is held already; within the
 * budget after, but for one bigger than all of it. */
bool
insert(const cache_key &k, std::string &&wgsl, bool fresh)
{
   read_budget();
   if (wgsl.size() > cache_budget || cache.count(k))
      return false;
   cache_lru.push_front(k);
   cache_bytes += wgsl.size();
   cache.emplace(k, cache_entry{std::move(wgsl), fresh, cache_lru.begin()});
   evict_to(cache_budget);
   return true;
}

const char cache_magic[8] = {'G', 'X', 'W', 'G', 'S', 'L', '1', 0};
/* This build: a blob from any other is not trusted. */
const char cache_build[] = __DATE__ " " __TIME__ " " __FILE__;

cache_key key_of(const uint32_t *words, size_t n)
{
   uint64_t a = 0xcbf29ce484222325ull ^ n, b = 0x84222325cbf29ce4ull + n;
   for (size_t i = 0; i < n; ++i) {
      a = (a ^ words[i]) * 0x100000001b3ull;
      b += words[i] + 0x9e3779b97f4a7c15ull;
      b = (b ^ (b >> 30)) * 0xbf58476d1ce4e5b9ull;
      b = (b ^ (b >> 27)) * 0x94d049bb133111ebull;
   }
   return {a, b ^ (b >> 31)};
}

uint8_t *put(uint8_t *at, const void *v, size_t n)
{
   memcpy(at, v, n);
   return at + n;
}

} // namespace

extern "C" char *
gex_wgpu_spirv_to_wgsl(const uint32_t *words, size_t word_count,
                       char **error_out)
{
   const char *off = getenv("GEX_WGSL_CACHE");
   if ((off && !strcmp(off, "0")) || !words || !word_count)
      return translate(words, word_count, error_out);
   const cache_key k = key_of(words, word_count);
   {
      std::lock_guard<std::mutex> lock(cache_mutex);
      auto it = cache.find(k);
      if (it != cache.end()) {
         ++cache_hits;
         cache_lru.splice(cache_lru.begin(), cache_lru, it->second.lru);
         if (error_out)
            *error_out = nullptr;
         return copy_string(it->second.wgsl);
      }
      ++cache_misses;
   }
   char *wgsl = translate(words, word_count, error_out);
   if (wgsl) {
      std::lock_guard<std::mutex> lock(cache_mutex);
      insert(k, std::string(wgsl), true);
   }
   return wgsl;
}

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define GEX_WGSL_EXPORT extern "C" EMSCRIPTEN_KEEPALIVE __attribute__((visibility("default")))
#else
#define GEX_WGSL_EXPORT extern "C" __attribute__((visibility("default")))
#endif

/* Entries from a blob `gex_wgsl_cache_dump` made: how many were added, or -1
 * if the blob is not this build's (or is malformed; nothing is added then). */
GEX_WGSL_EXPORT int
gex_wgsl_cache_load(const uint8_t *blob, uint32_t len)
{
   const uint32_t idlen = sizeof(cache_build);
   if (!blob || len < 12 + idlen || memcmp(blob, cache_magic, 8))
      return -1;
   uint32_t got;
   memcpy(&got, blob + 8, 4);
   if (got != idlen || memcmp(blob + 12, cache_build, idlen))
      return -1;
   /* Checked whole first, so that a bad one adds nothing, and then read
    * straight into the cache: not into a copy of all of it first. */
   for (uint32_t at = 12 + idlen; at < len;) {
      uint32_t n;
      if (len - at < 20)
         return -1;
      memcpy(&n, blob + at + 16, 4);
      at += 20;
      if (len - at < n)
         return -1;
      at += n;
   }
   std::lock_guard<std::mutex> lock(cache_mutex);
   int added = 0;
   for (uint32_t at = 12 + idlen; at < len;) {
      cache_key k;
      uint32_t n;
      memcpy(&k.a, blob + at, 8);
      memcpy(&k.b, blob + at + 8, 8);
      memcpy(&n, blob + at + 16, 4);
      at += 20;
      /* Behind what is already held: the budget takes the blob's own
       * first, which nothing has asked for yet. */
      read_budget();
      if (n <= cache_budget && !cache.count(k)) {
         cache_lru.push_back(k);
         cache_bytes += n;
         cache.emplace(k, cache_entry{std::string(reinterpret_cast<const char *>(blob + at), n), false,
                                      std::prev(cache_lru.end())});
         ++added;
      }
      at += n;
   }
   evict_to(cache_budget);
   cache_loaded += added;
   return added;
}

/* The most bytes of WGSL the cache keeps; 0 keeps none. What is over goes
 * at once, least recently used first. */
GEX_WGSL_EXPORT void
gex_wgsl_cache_budget(uint32_t bytes)
{
   std::lock_guard<std::mutex> lock(cache_mutex);
   cache_budget_read = true;
   cache_budget = bytes;
   evict_to(cache_budget);
}

/* The cache as a blob (malloc'd; the caller frees it), its length in
 * *len_out: every entry, or with `only_fresh` only those translated in this
 * process since it started. */
GEX_WGSL_EXPORT uint8_t *
gex_wgsl_cache_dump(int only_fresh, uint32_t *len_out)
{
   std::lock_guard<std::mutex> lock(cache_mutex);
   /* Measured, then written into the one allocation: not built up in a
    * string and copied, which held it twice. */
   const uint32_t idlen = sizeof(cache_build);
   size_t size = 12 + idlen;
   for (const auto &e : cache) {
      if (!only_fresh || e.second.fresh)
         size += 20 + e.second.wgsl.size();
   }
   uint8_t *blob = static_cast<uint8_t *>(std::malloc(size));
   if (!blob)
      return nullptr;
   uint8_t *at = put(blob, cache_magic, 8);
   at = put(at, &idlen, 4);
   at = put(at, cache_build, idlen);
   for (const auto &e : cache) {
      if (only_fresh && !e.second.fresh)
         continue;
      const uint32_t n = (uint32_t)e.second.wgsl.size();
      at = put(at, &e.first.a, 8);
      at = put(at, &e.first.b, 8);
      at = put(at, &n, 4);
      at = put(at, e.second.wgsl.data(), n);
   }
   if (len_out)
      *len_out = (uint32_t)size;
   return blob;
}

/* hits, misses, entries loaded from blobs, entries now held. */
GEX_WGSL_EXPORT void
gex_wgsl_cache_stats(uint32_t *out)
{
   std::lock_guard<std::mutex> lock(cache_mutex);
   out[0] = cache_hits;
   out[1] = cache_misses;
   out[2] = cache_loaded;
   out[3] = (uint32_t)cache.size();
}

/* The same, and entries evicted and bytes of WGSL held: `out` of six. */
GEX_WGSL_EXPORT void
gex_wgsl_cache_stats2(uint32_t *out)
{
   gex_wgsl_cache_stats(out);
   std::lock_guard<std::mutex> lock(cache_mutex);
   out[4] = cache_evicted;
   out[5] = (uint32_t)std::min(cache_bytes, (size_t)UINT32_MAX);
}
