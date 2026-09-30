/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * 64-bit atomics on buffers and workgroup memory (shaderBufferInt64Atomics,
 * shaderSharedInt64Atomics), which WGSL has none of: each under a lock of
 * its own words' -- one of a table, by the address -- taken, the words read
 * and written, and given back in one turn of a loop, so that invocations of
 * one SIMD group, which a lock taken in one turn and given back in the next
 * would leave spinning on each other, do not wait on each other. The words
 * are read and written by 32-bit atomics too, which see each other's in
 * order whatever the lock's order is to them.
 *
 * A buffer's locks are the device's (WVK_DRIVER_SLOT_LOCKS); workgroup
 * memory's are after the workgroup's own, zeroed at the start.
 */
#include "wvk_nir.h"

#include "wvk_descriptor_set_layout.h"

#include "compiler/nir/nir_builder.h"

/* The workgroup's locks. */
#define SHARED_LOCKS 16

static bool
is_atomic64(const nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
      return intr->def.bit_size == 64;
   default:
      return false;
   }
}

static nir_def *
locks_buffer(nir_builder *b)
{
   nir_def *index = nir_vulkan_resource_index(b, 2, 32, nir_imm_int(b, 0), .desc_set = WVK_DRIVER_GROUP,
                                              .binding = WVK_DRIVER_SLOT_LOCKS,
                                              .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   nir_def *desc = nir_load_vulkan_descriptor(b, 2, 32, index, .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   return nir_channel(b, desc, 0);
}

struct place {
   bool shared;
   nir_def *buffer;
};

/* A 32-bit atomic of `op` at `offset` of the place. */
static nir_def *
word_atomic(nir_builder *b, const struct place *p, nir_def *offset, nir_def *data, nir_atomic_op op)
{
   if (p->shared)
      return nir_shared_atomic(b, 32, offset, data, .atomic_op = op);
   return nir_ssbo_atomic(b, 32, p->buffer, offset, data, .atomic_op = op);
}

static nir_def *
word_cmpxchg(nir_builder *b, const struct place *p, nir_def *offset, nir_def *compare, nir_def *data)
{
   if (p->shared)
      return nir_shared_atomic_swap(b, 32, offset, compare, data, .atomic_op = nir_atomic_op_cmpxchg);
   return nir_ssbo_atomic_swap(b, 32, p->buffer, offset, compare, data, .atomic_op = nir_atomic_op_cmpxchg);
}

static void
lower_atomic(nir_builder *b, nir_intrinsic_instr *intr, uint32_t shared_locks)
{
   const bool shared =
      intr->intrinsic == nir_intrinsic_shared_atomic || intr->intrinsic == nir_intrinsic_shared_atomic_swap;
   const bool swap =
      intr->intrinsic == nir_intrinsic_ssbo_atomic_swap || intr->intrinsic == nir_intrinsic_shared_atomic_swap;
   const unsigned first = shared ? 0 : 1;
   const nir_atomic_op op = nir_intrinsic_atomic_op(intr);
   b->cursor = nir_before_instr(&intr->instr);

   const struct place data_place = { shared, shared ? NULL : intr->src[0].ssa };
   nir_def *offset = intr->src[first].ssa;
   if (shared)
      offset = nir_iadd_imm(b, offset, nir_intrinsic_base(intr));
   nir_def *data = intr->src[first + 1].ssa;
   nir_def *swap_data = swap ? intr->src[first + 2].ssa : NULL;

   /* Its lock: of its eight bytes' place in the table. */
   const struct place device_locks = { false, locks_buffer(b) };
   const struct place lock_place = shared ? (struct place){ true, NULL } : device_locks;
   const unsigned locks = shared ? SHARED_LOCKS : WVK_ATOMIC_LOCKS;
   nir_def *lock = nir_ishl_imm(b, nir_iand_imm(b, nir_ushr_imm(b, offset, 3), locks - 1), 2);
   if (shared)
      lock = nir_iadd_imm(b, lock, shared_locks);
   /* A zero no compiler knows is one, the word after the device's locks':
    * what each access is made to depend on of the one before it. Relaxed
    * atomics of different addresses, which all of WGSL's are, may be
    * reordered, and the words be written after the lock is given back,
    * or read before it is taken -- an exchange's, whose value is not of
    * what it reads, were. */
   nir_def *zero = word_atomic(b, &device_locks, nir_imm_int(b, WVK_ATOMIC_LOCKS * 4), nir_imm_int(b, 0),
                               nir_atomic_op_ior);

   /* Its words: no 64-bit variable, which the 64-bit lowering has none of. */
   nir_variable *result = nir_local_variable_create(b->impl, glsl_uvec2_type(), "wvk_atomic64");
   nir_loop *loop = nir_push_loop(b);
   {
      nir_def *held = word_cmpxchg(b, &lock_place, lock, nir_imm_int(b, 0), nir_imm_int(b, 1));
      nir_def *after_lock = nir_iand(b, held, zero);
      nir_push_if(b, nir_ieq_imm(b, held, 0));
      {
         nir_def *at = nir_iadd(b, offset, after_lock);
         nir_def *lo = word_atomic(b, &data_place, at, nir_imm_int(b, 0), nir_atomic_op_iadd);
         nir_def *hi = word_atomic(b, &data_place, nir_iadd_imm(b, at, 4), nir_imm_int(b, 0), nir_atomic_op_iadd);
         nir_def *old = nir_pack_64_2x32_split(b, lo, hi);
         nir_def *value;
         switch (op) {
         case nir_atomic_op_xchg:
            value = data;
            break;
         case nir_atomic_op_cmpxchg:
            value = nir_bcsel(b, nir_ieq(b, old, data), swap_data, old);
            break;
         default:
            value = nir_build_alu2(b, nir_atomic_op_to_alu(op), old, data);
            break;
         }
         nir_def *after_read = nir_iand(b, nir_ior(b, lo, hi), zero);
         nir_def *new_lo = nir_ior(b, nir_unpack_64_2x32_split_x(b, value), after_read);
         nir_def *new_hi = nir_ior(b, nir_unpack_64_2x32_split_y(b, value), after_read);
         nir_def *was_lo = word_atomic(b, &data_place, at, new_lo, nir_atomic_op_xchg);
         nir_def *was_hi = word_atomic(b, &data_place, nir_iadd_imm(b, at, 4), new_hi, nir_atomic_op_xchg);
         word_atomic(b, &lock_place, nir_iadd(b, lock, nir_iand(b, nir_ior(b, was_lo, was_hi), zero)),
                     nir_imm_int(b, 0), nir_atomic_op_xchg);
         nir_store_var(b, result, nir_vec2(b, lo, hi), 0x3);
         nir_jump(b, nir_jump_break);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_loop(b, loop);
   nir_def_replace(&intr->def, nir_pack_64_2x32(b, nir_load_var(b, result)));
}

bool
wvk_nir_lower_atomic64(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct util_dynarray atomics;
   util_dynarray_init(&atomics, NULL);
   bool any_shared = false;
   nir_foreach_function_impl(fi, nir) {
      nir_foreach_block(block, fi) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic || !is_atomic64(nir_instr_as_intrinsic(instr)))
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            util_dynarray_append(&atomics, intr);
            any_shared |= intr->intrinsic == nir_intrinsic_shared_atomic ||
                          intr->intrinsic == nir_intrinsic_shared_atomic_swap;
         }
      }
   }
   if (!atomics.size) {
      util_dynarray_fini(&atomics);
      return false;
   }

   /* The workgroup's locks after its memory, zeroed first by the
    * invocations between them. */
   uint32_t shared_locks = 0;
   if (any_shared) {
      shared_locks = align(nir->info.shared_size, 4);
      nir->info.shared_size = shared_locks + SHARED_LOCKS * 4;
      nir_builder b = nir_builder_at(nir_before_impl(impl));
      const unsigned invocations =
         MAX2(nir->info.workgroup_size[0] * nir->info.workgroup_size[1] * nir->info.workgroup_size[2], 1);
      nir_def *index = nir_load_local_invocation_index(&b);
      for (unsigned i = 0; i < SHARED_LOCKS; i += invocations) {
         nir_def *at = nir_iadd_imm(&b, index, i);
         nir_push_if(&b, nir_ult_imm(&b, at, SHARED_LOCKS));
         nir_store_shared(&b, nir_imm_int(&b, 0), nir_iadd_imm(&b, nir_ishl_imm(&b, at, 2), shared_locks),
                          .write_mask = 0x1, .align_mul = 4);
         nir_pop_if(&b, NULL);
      }
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   }

   nir_builder b = nir_builder_create(impl);
   util_dynarray_foreach(&atomics, nir_intrinsic_instr *, intr) {
      b.impl = nir_cf_node_get_function(&(*intr)->instr.block->cf_node);
      lower_atomic(&b, *intr, shared_locks);
   }
   util_dynarray_fini(&atomics);
   nir_foreach_function_impl(fi, nir)
      nir_progress(true, fi, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   return true;
}
