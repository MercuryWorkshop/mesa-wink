/* SPDX-License-Identifier: MIT */
/*
 * st/mesa options from Gallium's driconf defaults, as DRI and WGL get them.
 *
 * gex's decoder handed st_api_create_context zeroed st_config_options, so the
 * browser ran with every option off while a native DRI build of the same
 * driver ran with the defaults on -- allow_compressed_fallback among them,
 * which is what lets st/mesa expose ASTC (and with it GLES 3.2) by decoding
 * on the CPU. Taking the same table here makes a native CTS run a prediction
 * of what the browser does rather than of something else.
 *
 * Built with xmlconfig disabled there are no drirc files to read, but
 * driParseConfigFiles still applies environment overrides
 * (allow_compressed_fallback=false in Module.winkEnv, say), and
 * MESA_DRICONF_EXECUTABLE_OVERRIDE names the application for its built-in
 * workaround table, a browser module having no executable name of its own.
 */
#include "wink_private.h"

#include "frontend/api.h"
#include "util/driconf.h"
#include "util/u_driconf.h"
#include "util/xmlconfig.h"

static const driOptionDescription wink_driconf[] = {
#include "pipe-loader/driinfo_gallium.h"
};

void
wink_fill_st_options(struct st_config_options *options)
{
   driOptionCache info, cache;
   driParseOptionInfo(&info, wink_driconf, ARRAY_SIZE(wink_driconf));
   driParseConfigFiles(&cache, &info, &(driConfigFileParseParams) {
      .driverName = "gex_wgpu",
   });
   u_driconf_fill_st_options(options, &cache);
   driDestroyOptionCache(&cache);
   driDestroyOptionInfo(&info);
}
