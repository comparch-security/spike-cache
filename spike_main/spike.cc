// See LICENSE for license details.

#include "sim.h"
#include "mmu.h"
#include "remote_bitbang.h"
#include "cachesim.h"
#include "extension.h"
#include <dlfcn.h>
#include <fesvr/option_parser.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <memory>
#include "cache/cache.hpp"
#include "util/cache_config_parser.hpp"
#include "util/report.hpp"
#include "../VERSION"
extern Reporter_t reporter;

static void help(int exit_code = 1)
{
  fprintf(stderr, "Spike RISC-V ISA Simulator " SPIKE_VERSION "\n\n");
  fprintf(stderr, "usage: spike [host options] <target program> [target options]\n");
  fprintf(stderr, "Host Options:\n");
  fprintf(stderr, "  -p<n>                 Simulate <n> processors [default 1]\n");
  fprintf(stderr, "  -m<n>                 Provide <n> MiB of target memory [default 2048]\n");
  fprintf(stderr, "  -m<a:m,b:n,...>       Provide memory regions of size m and n bytes\n");
  fprintf(stderr, "                          at base addresses a and b (with 4 KiB alignment)\n");
  fprintf(stderr, "  -d                    Interactive debug mode\n");
  fprintf(stderr, "  -g                    Track histogram of PCs\n");
  fprintf(stderr, "  -l                    Generate a log of execution\n");
  fprintf(stderr, "  -h, --help            Print this help message\n");
  fprintf(stderr, "  -H                    Start halted, allowing a debugger to connect\n");
  fprintf(stderr, "  --isa=<name>          RISC-V ISA string [default %s]\n", DEFAULT_ISA);
  fprintf(stderr, "  --varch=<name>        RISC-V Vector uArch string [default %s]\n", DEFAULT_VARCH);
  fprintf(stderr, "  --pc=<address>        Override ELF entry point\n");
  fprintf(stderr, "  --hartids=<a,b,...>   Explicitly specify hartids, default is 0,1,...\n");
  fprintf(stderr, "  --cache-cfile=<name>  the configuration file for the cache model [default cache_model/config/cache.json]\n");
  fprintf(stderr, "  --cache-model=<name>  name of the cache mod\n");
  fprintf(stderr, "  --extension=<name>    Specify RoCC Extension\n");
  fprintf(stderr, "  --extlib=<name>       Shared library to load\n");
  fprintf(stderr, "  --rbb-port=<port>     Listen on <port> for remote bitbang connection\n");
  fprintf(stderr, "  --trace               trace the cache activity\n");
  fprintf(stderr, "  --dump-dts            Print device tree string and exit\n");
  fprintf(stderr, "  --disable-dtb         Don't write the device tree blob into memory\n");
  fprintf(stderr, "  --dm-progsize=<words> Progsize for the debug module [default 2]\n");
  fprintf(stderr, "  --dm-sba=<bits>       Debug bus master supports up to "
      "<bits> wide accesses [default 0]\n");
  fprintf(stderr, "  --dm-auth             Debug module requires debugger to authenticate\n");
  fprintf(stderr, "  --dmi-rti=<n>         Number of Run-Test/Idle cycles "
      "required for a DMI access [default 0]\n");
  fprintf(stderr, "  --dm-abstract-rti=<n> Number of Run-Test/Idle cycles "
      "required for an abstract command to execute [default 0]\n");
  fprintf(stderr, "  --dm-no-hasel         Debug module supports hasel\n");
  fprintf(stderr, "  --dm-no-abstract-csr  Debug module won't support abstract to authenticate\n");
  fprintf(stderr, "  --dm-no-halt-groups   Debug module won't support halt groups\n");

  exit(exit_code);
}

static void suggest_help()
{
  fprintf(stderr, "Try 'spike --help' for more information.\n");
  exit(1);
}

static std::vector<std::pair<reg_t, mem_t*>> make_mems(const char* arg)
{
  // handle legacy mem argument
  char* p;
  auto mb = strtoull(arg, &p, 0);
  if (*p == 0) {
    reg_t size = reg_t(mb) << 20;
    if (size != (size_t)size)
      throw std::runtime_error("Size would overflow size_t");
    return std::vector<std::pair<reg_t, mem_t*>>(1, std::make_pair(reg_t(DRAM_BASE), new mem_t(size)));
  }

  // handle base/size tuples
  std::vector<std::pair<reg_t, mem_t*>> res;
  while (true) {
    auto base = strtoull(arg, &p, 0);
    if (!*p || *p != ':')
      help();
    auto size = strtoull(p + 1, &p, 0);
    if ((size | base) % PGSIZE != 0)
      help();
    res.push_back(std::make_pair(reg_t(base), new mem_t(size)));
    if (!*p)
      break;
    if (*p != ',')
      help();
    arg = p + 1;
  }
  return res;
}

int main(int argc, char** argv)
{
  bool debug = false;
  bool halted = false;
  bool histogram = false;
  bool log = false;
  bool trace = false;
  bool dump_dts = false;
  bool dtb_enabled = true;
  size_t nprocs = 1;
  reg_t start_pc = reg_t(-1);
  std::vector<std::pair<reg_t, mem_t*>> mems;
  std::string cache_cfg_file("cache_model/config/cache.json");
  std::string cache_cfg_model;
  CacheCFG ccfg;
  std::vector<CoherentCache *> l1_caches;
  std::vector<CoherentCache *> l2_caches;
  std::function<extension_t*()> extension;
  const char* isa = DEFAULT_ISA;
  const char* varch = DEFAULT_VARCH;
  uint16_t rbb_port = 0;
  bool use_rbb = false;
  unsigned dmi_rti = 0;
  debug_module_config_t dm_config = {
    .progbufsize = 2,
    .max_bus_master_bits = 0,
    .require_authentication = false,
    .abstract_rti = 0,
    .support_hasel = true,
    .support_abstract_csr_access = true,
    .support_haltgroups = true
  };
  std::vector<int> hartids;

  auto const hartids_parser = [&](const char *s) {
    std::string const str(s);
    std::stringstream stream(str);

    int n;
    while (stream >> n)
    {
      hartids.push_back(n);
      if (stream.peek() == ',') stream.ignore();
    }
  };

  option_parser_t parser;
  parser.help(&suggest_help);
  parser.option('h', "help", 0, [&](const char* s){help(0);});
  parser.option('d', 0, 0, [&](const char* s){debug = true;});
  parser.option('g', 0, 0, [&](const char* s){histogram = true;});
  parser.option('l', 0, 0, [&](const char* s){log = true;});
  parser.option('p', 0, 1, [&](const char* s){nprocs = atoi(s);});
  parser.option('m', 0, 1, [&](const char* s){mems = make_mems(s);});
  // I wanted to use --halted, but for some reason that doesn't work.
  parser.option('H', 0, 0, [&](const char* s){halted = true;});
  parser.option(0, "rbb-port", 1, [&](const char* s){use_rbb = true; rbb_port = atoi(s);});
  parser.option(0, "pc", 1, [&](const char* s){start_pc = strtoull(s, 0, 0);});
  parser.option(0, "hartids", 1, hartids_parser);
  parser.option(0, "cache-cfile", 1, [&](const char* s){cache_cfg_file = std::string(s);});
  parser.option(0, "cache-model", 1, [&](const char* s){cache_cfg_model = std::string(s);});
  parser.option(0, "isa", 1, [&](const char* s){isa = s;});
  parser.option(0, "varch", 1, [&](const char* s){varch = s;});
  parser.option(0, "extension", 1, [&](const char* s){extension = find_extension(s);});
  parser.option(0, "trace", 0, [&](const char *s){trace = true;});
  parser.option(0, "dump-dts", 0, [&](const char *s){dump_dts = true;});
  parser.option(0, "disable-dtb", 0, [&](const char *s){dtb_enabled = false;});
  parser.option(0, "extlib", 1, [&](const char *s){
    void *lib = dlopen(s, RTLD_NOW | RTLD_GLOBAL);
    if (lib == NULL) {
      fprintf(stderr, "Unable to load extlib '%s': %s\n", s, dlerror());
      exit(-1);
    }
  });
  parser.option(0, "dm-progsize", 1,
      [&](const char* s){dm_config.progbufsize = atoi(s);});
  parser.option(0, "dm-sba", 1,
      [&](const char* s){dm_config.max_bus_master_bits = atoi(s);});
  parser.option(0, "dm-auth", 0,
      [&](const char* s){dm_config.require_authentication = true;});
  parser.option(0, "dmi-rti", 1,
      [&](const char* s){dmi_rti = atoi(s);});
  parser.option(0, "dm-abstract-rti", 1,
      [&](const char* s){dm_config.abstract_rti = atoi(s);});
  parser.option(0, "dm-no-hasel", 0,
      [&](const char* s){dm_config.support_hasel = false;});
  parser.option(0, "dm-no-abstract-csr", 0,
      [&](const char* s){dm_config.support_abstract_csr_access = false;});
  parser.option(0, "dm-no-halt-groups", 0,
      [&](const char* s){dm_config.support_haltgroups = false;});

  auto argv1 = parser.parse(argv);
  std::vector<std::string> htif_args(argv1, (const char*const*)argv + argc);
  if (mems.empty())
    mems = make_mems("2048");

  if (!*argv1)
    help();

  sim_t s(isa, varch, nprocs, halted, start_pc, mems, htif_args, std::move(hartids),
      dm_config);
  std::unique_ptr<remote_bitbang_t> remote_bitbang((remote_bitbang_t *) NULL);
  std::unique_ptr<jtag_dtm_t> jtag_dtm(
      new jtag_dtm_t(&s.debug_module, dmi_rti));
  if (use_rbb) {
    remote_bitbang.reset(new remote_bitbang_t(rbb_port, &(*jtag_dtm)));
    s.set_remote_bitbang(&(*remote_bitbang));
  }
  s.set_dtb_enabled(dtb_enabled);

  if (dump_dts) {
    printf("%s", s.get_dts());
    return 0;
  }

  // cache model
  if(!cache_cfg_model.empty()) {
    if(!cache_config_parser(cache_cfg_file, cache_cfg_model, &ccfg)) exit(1);
    if(ccfg.enable[0] && ccfg.number[0] != nprocs * 2) {
      fprintf(stderr, "The number of L1 caches %d do not match with the number of cores %ld\n", ccfg.number[0], nprocs);
      exit(1);
    }

    if(ccfg.enable[0]) l1_caches.resize(ccfg.number[0]);
    if(ccfg.enable[1]) l2_caches.resize(ccfg.number[1]);

    if(ccfg.enable[0]) {
      for(unsigned int i=0; i<ccfg.number[0]; i++)
        l1_caches[i] = new L1CacheBase(i, i/2, i%2, ccfg.cache_gen[0],
                                       ccfg.enable[1] ? &l2_caches : NULL,
                                       ccfg.hash_gen[0]);
    }

    if(ccfg.enable[1]) {
      for(unsigned int i=0; i<ccfg.number[1]; i++) {
        l2_caches[i] = new LLCCacheBase(i, 2, ccfg.cache_gen[1], &l1_caches);
      }
    }

    if(trace) {
      for(unsigned int i=0; i<ccfg.number[0]; i++) {
        reporter.register_cache_access_tracer(1, i/2, (int)i%2);
      }
      reporter.register_cache_access_tracer(2);
    }

    if(ccfg.enable[0])
      for(size_t i = 0; i < nprocs; i++) {
        s.get_core(i)->get_mmu()->register_cache_models(l1_caches[2*i], l1_caches[2*i+1]);
      }
  }

  for (size_t i = 0; i < nprocs; i++)
  {
    if (extension) s.get_core(i)->register_extension(extension());
  }

  s.set_debug(debug);
  s.set_log(log);
  s.set_histogram(histogram);
  bool rv = s.run();

  if(trace) {
    fprintf(stdout, "------- statistics --------\n");
    for(size_t i = 0; i < nprocs; i++) {
      fprintf(stdout, "Core %ld runs %lu instructions\n", i, s.get_core(i)->get_state()->minstret);
    }
    for(unsigned int i=0; i<ccfg.number[0]; i++) {
      uint64_t miss = reporter.check_cache_miss(1, i/2, i%2);
      uint64_t evict = reporter.check_cache_evict(1, i/2, i%2);
      uint64_t writeback = reporter.check_cache_writeback(1, i/2, i%2);
      uint64_t access = reporter.check_cache_access(1, i/2, i%2);
      fprintf(stdout, "%s: miss %lu, evict %lu, and writeback %lu in %lu access, miss rate %f\n", l1_caches[i]->cache_name().c_str(), miss, evict, writeback, access, (float)(miss)/access);
    }
    uint64_t miss = reporter.check_cache_miss(2);
    uint64_t evict = reporter.check_cache_evict(2);
    uint64_t writeback = reporter.check_cache_writeback(2);
    uint64_t access = reporter.check_cache_access(2);
    fprintf(stdout, "L2: miss %lu, evict %lu and writeback %lu in %lu access, miss rate %f\n", miss, evict, writeback, access, (float)(miss)/access);
  }

  return rv;
}
