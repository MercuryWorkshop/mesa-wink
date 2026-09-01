/* SPDX-License-Identifier: MIT
 * Isolated C ABI bridge to Tint.  Shader semantics are transformed in NIR;
 * this translation step only converts valid SPIR-V to valid WGSL.
 */
#include "gex_wgpu_tint.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "src/tint/api/tint.h"
#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/writer/writer.h"

namespace {

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
   tint::spirv::reader::Options options;
   options.allow_non_uniform_derivatives = true;
   options.allowed_features = tint::wgsl::AllowedFeatures::Everything();
   tint::Program program = tint::spirv::reader::Read(spirv, options);
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
   char *wgsl = copy_string(generated.Get().wgsl);
   if (!wgsl)
      set_error(error_out, "out of memory copying WGSL");
   return wgsl;
}
