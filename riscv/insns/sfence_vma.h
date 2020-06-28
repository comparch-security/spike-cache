require_privilege(get_field(STATE.mstatus, MSTATUS_TVM) ? PRV_M : PRV_S);
MMU.flush_tlb();
// this is a simplified implementation ignoring rs1 and rs2
// See privileged spec 4.2.1:
//   Supervisor Memory-Management Fence Instruction
MMU.flush_hard_tlb_d();
MMU.flush_hard_tlb_i();
