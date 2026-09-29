/* SPDX-License-Identifier: MIT
 * Isolated C ABI bridge to Tint.  Shader semantics are transformed in NIR;
 * this translation step only converts valid SPIR-V to valid WGSL.
 */
#include "gex_wgpu_tint.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/tint/api/tint.h"
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

extern "C" char *
gex_wgpu_spirv_to_wgsl(const uint32_t *words, size_t word_count,
                       char **error_out)
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
   if (!program.IsValid() && only_barrier_uniformity(program.Diagnostics())) {
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
