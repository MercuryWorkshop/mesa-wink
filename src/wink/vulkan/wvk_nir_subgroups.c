/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Subgroups of one invocation. Tint reads no subgroup operations from
 * SPIR-V, and WGSL has them only in compute and fragment shaders, so
 * every invocation is a subgroup of its own: each operation is what it
 * is over one invocation, in every stage. Quad operations, which need
 * four, are not supported.
 */
#include "wvk_nir.h"

#include "compiler/nir/nir_builder.h"

/* A ballot of the invocation alone, or of none: bit 0 of the first
 * component, in the shape the intrinsic returns. */
static nir_def *
ballot_of(nir_builder *b, nir_def *bit, unsigned num_components, unsigned bit_size)
{
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   comps[0] = nir_u2uN(b, bit, bit_size);
   for (unsigned i = 1; i < num_components; ++i)
      comps[i] = nir_imm_intN_t(b, 0, bit_size);
   return nir_vec(b, comps, num_components);
}

/* Bit 0 of a ballot, as a 32-bit 0 or 1. */
static nir_def *
ballot_bit0(nir_builder *b, nir_def *ballot)
{
   return nir_u2u32(b, nir_iand_imm(b, nir_channel(b, ballot, 0), 1));
}

static bool
lower_subgroup_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *def = nir_intrinsic_infos[intr->intrinsic].has_dest ? &intr->def : NULL;
   nir_def *value;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_subgroup_size:
      value = nir_imm_int(b, 1);
      break;
   case nir_intrinsic_load_subgroup_invocation:
   case nir_intrinsic_first_invocation:
   case nir_intrinsic_last_invocation:
      value = nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_num_subgroups: {
      nir_def *size = nir_load_workgroup_size(b);
      value = nir_imul(b, nir_imul(b, nir_channel(b, size, 0), nir_channel(b, size, 1)), nir_channel(b, size, 2));
      break;
   }
   case nir_intrinsic_load_subgroup_id:
      value = nir_load_local_invocation_index(b);
      break;
   /* Only invocation 0 is in the subgroup: it is equal, and at most and
    * at least itself. */
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_le_mask:
      value = ballot_of(b, nir_imm_int(b, 1), def->num_components, def->bit_size);
      break;
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
      value = ballot_of(b, nir_imm_int(b, 0), def->num_components, def->bit_size);
      break;
   case nir_intrinsic_ballot:
   case nir_intrinsic_ballot_relaxed:
      value = ballot_of(b, nir_b2i32(b, intr->src[0].ssa), def->num_components, def->bit_size);
      break;
   case nir_intrinsic_inverse_ballot:
      value = nir_ine_imm(b, ballot_bit0(b, intr->src[0].ssa), 0);
      break;
   case nir_intrinsic_ballot_bitfield_extract: {
      /* Any bit of the value, not only the invocation's. */
      nir_def *ballot = intr->src[0].ssa;
      nir_def *index = nir_u2u32(b, intr->src[1].ssa);
      const unsigned bits = ballot->bit_size;
      nir_def *comp = nir_vector_extract(b, ballot, nir_udiv_imm(b, index, bits));
      nir_def *bit = nir_ushr(b, comp, nir_umod_imm(b, index, bits));
      value = nir_ine_imm(b, nir_iand_imm(b, bit, 1), 0);
      break;
   }
   /* Counts and searches are of the subgroup's bits: bit 0. */
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_ballot_bit_count_inclusive:
      value = ballot_bit0(b, intr->src[0].ssa);
      break;
   case nir_intrinsic_ballot_bit_count_exclusive:
      value = nir_imm_int(b, 0);
      break;
   case nir_intrinsic_ballot_find_lsb:
      value = nir_find_lsb(b, ballot_bit0(b, intr->src[0].ssa));
      break;
   case nir_intrinsic_ballot_find_msb:
      value = nir_ufind_msb(b, ballot_bit0(b, intr->src[0].ssa));
      break;
   case nir_intrinsic_elect:
   case nir_intrinsic_vote_feq:
   case nir_intrinsic_vote_ieq:
      value = nir_imm_true(b);
      break;
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_quad_vote_any:
   case nir_intrinsic_quad_vote_all:
   /* Every invocation read is this one, or undefined. */
   case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_shuffle:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_rotate:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
   case nir_intrinsic_reduce:
   case nir_intrinsic_inclusive_scan:
      value = intr->src[0].ssa;
      break;
   case nir_intrinsic_exclusive_scan: {
      /* Of no invocations: the operation's identity. */
      const nir_op op = nir_intrinsic_reduction_op(intr);
      const nir_const_value identity = nir_alu_binop_identity(op, def->bit_size);
      nir_const_value values[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < def->num_components; ++i)
         values[i] = identity;
      value = nir_build_imm(b, def->num_components, def->bit_size, values);
      break;
   }
   case nir_intrinsic_barrier: {
      /* One invocation is in step with itself, and sees its own writes. */
      bool changed = false;
      if (nir_intrinsic_execution_scope(intr) == SCOPE_SUBGROUP) {
         nir_intrinsic_set_execution_scope(intr, SCOPE_NONE);
         changed = true;
      }
      if (nir_intrinsic_memory_scope(intr) == SCOPE_SUBGROUP ||
          nir_intrinsic_memory_scope(intr) == SCOPE_INVOCATION) {
         nir_intrinsic_set_memory_scope(intr, SCOPE_NONE);
         nir_intrinsic_set_memory_semantics(intr, 0);
         nir_intrinsic_set_memory_modes(intr, 0);
         changed = true;
      }
      if (nir_intrinsic_execution_scope(intr) == SCOPE_NONE && nir_intrinsic_memory_scope(intr) == SCOPE_NONE) {
         nir_instr_remove(&intr->instr);
         return true;
      }
      /* An execution barrier with no memory is WGSL's workgroupBarrier()
       * still: Tint makes nothing of a control barrier without memory
       * semantics, and SPIR-V has no scope for none. */
      if (nir_intrinsic_execution_scope(intr) != SCOPE_NONE && nir_intrinsic_memory_scope(intr) == SCOPE_NONE) {
         nir_intrinsic_set_memory_scope(intr, SCOPE_WORKGROUP);
         nir_intrinsic_set_memory_semantics(intr, NIR_MEMORY_ACQ_REL);
         nir_intrinsic_set_memory_modes(intr, nir_var_mem_shared);
         changed = true;
      }
      return changed;
   }
   default:
      return false;
   }

   nir_def_replace(def, value);
   return true;
}

bool
wvk_nir_lower_subgroups(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_subgroup_instr, nir_metadata_control_flow, NULL);
}
