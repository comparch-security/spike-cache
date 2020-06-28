// simulate the hardware TLB in actual hardware processors

#ifndef _RISCV_TLB_H
#define _TISCV_TLB_H

#include <map>
#include <vector>

struct HardTLBEntry {
  bool     va;
  uint64_t ppn;
  uint64_t pte;
  HardTLBEntry(): va(false), ppn(0), pte(0) {}
  HardTLBEntry(bool va, uint64_t ppn, uint64_t pte): va(va), ppn(ppn), pte(pte) {}
};

struct WalkRecord {
  int levels;
  uint64_t vpn;
  uint64_t ppn;
  uint64_t pte;
  uint64_t ptes[6]; // maximally walk 6 page table entries
};

class HardTLBBase
{
protected:
  std::vector<std::map<uint64_t, HardTLBEntry> > entries;
  std::vector<std::list<uint64_t> > order;
  uint32_t nset, nway;  // number of sets and ways
  mmu_t *mmu;           // the pe core MMU
  CoherentCache *cache; // the L1 cache

public:
  uint64_t access_n, miss_n, hit_n, walk_hit_n; // pfcs

 HardTLBBase(mmu_t *mmu, CoherentCache *cache, uint32_t nway, uint32_t nset = 1)
  : nset(nset), nway(nway),
    mmu(mmu), cache(cache),
    access_n(0), miss_n(0), hit_n(0), walk_hit_n(0)
  {
    entries.resize(nset);
    order.resize(nset);
  }
  HardTLBEntry translate(uint64_t *latency, uint64_t vpn, access_type type, reg_t mode);
  void refill(uint64_t vpn, uint64_t ppn, uint64_t pte);
  void shootdown(uint64_t vpn);
  void flush();
};

#endif
