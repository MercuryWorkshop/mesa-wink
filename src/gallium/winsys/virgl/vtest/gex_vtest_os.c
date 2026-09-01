/* SPDX-License-Identifier: MIT */
/* Where the vtest winsys's socket calls go in a browser: see gex_vtest_os.h.
 *
 * The page hands this module a `Module.gexVirgl` with four methods, which are
 * the same four the guest's own transport uses -- open a session, feed it
 * bytes, take the reply, close it. virglrenderer runs those commands
 * synchronously in this worker, so a read after a write always has its answer
 * and nothing here ever has to wait.
 */
#include "gex_vtest_os.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>

/* Ask the page for a renderer session. Negative when the page has no
 * renderer, which the winsys reports as a failed connect. */
EM_JS(int, gex_vtest_js_open, (void), {
  const v = Module.gexVirgl;
  return v ? v.open() : -1;
});

/* Feed the session `len` bytes of command stream. Returns what it took. */
EM_JS(int, gex_vtest_js_write, (int session, const void *buf, int len), {
  const v = Module.gexVirgl;
  if (!v) return -1;
  // The view is taken here rather than held: growing the heap detaches an
  // old one, and a module that grows between calls would write into nothing.
  return v.write(session, HEAPU8.subarray(buf, buf + len));
});

/* Copy up to `max` reply bytes back. Zero means the renderer had nothing,
 * which for this protocol is a reply that never came. */
EM_JS(int, gex_vtest_js_read, (int session, void *buf, int max), {
  const v = Module.gexVirgl;
  if (!v) return -1;
  return v.read(session, HEAPU8.subarray(buf, buf + max));
});

EM_JS(void, gex_vtest_js_close, (int session), {
  Module.gexVirgl?.close(session);
});
#else
static int gex_vtest_js_open(void) { return -1; }
static int gex_vtest_js_write(int s, const void *b, int n) { (void)s; (void)b; (void)n; return -1; }
static int gex_vtest_js_read(int s, void *b, int n) { (void)s; (void)b; (void)n; return -1; }
static void gex_vtest_js_close(int s) { (void)s; }
#endif

/* A descriptor is a real one -- /dev/null, opened for the number -- so it can
 * never collide with something the filesystem handed out. What it *means* is
 * the session beside it. One connection per screen and one screen per module,
 * so a handful is more than enough. */
#define GEX_VTEST_MAX 4
static struct {
   int fd;
   int session;
} gex_vtest_open_fds[GEX_VTEST_MAX] = {
   { -1, -1 }, { -1, -1 }, { -1, -1 }, { -1, -1 },
};

static int gex_vtest_slot(int fd)
{
   for (int i = 0; i < GEX_VTEST_MAX; i++)
      if (gex_vtest_open_fds[i].fd == fd)
         return i;
   return -1;
}

int gex_vtest_socket(int domain, int type, int protocol)
{
   (void)domain; (void)type; (void)protocol;
   for (int i = 0; i < GEX_VTEST_MAX; i++) {
      if (gex_vtest_open_fds[i].fd >= 0)
         continue;
      int fd = open("/dev/null", O_RDWR);
      if (fd < 0)
         return -1;
      gex_vtest_open_fds[i].fd = fd;
      gex_vtest_open_fds[i].session = -1;
      return fd;
   }
   errno = EMFILE;
   return -1;
}

int gex_vtest_connect(int fd, const struct sockaddr *addr, socklen_t len)
{
   /* Whatever name the winsys was going to connect to, there is one renderer
    * and the page knows where it is. */
   (void)addr; (void)len;
   int i = gex_vtest_slot(fd);
   if (i < 0) {
      errno = EBADF;
      return -1;
   }
   int session = gex_vtest_js_open();
   if (session < 0) {
      errno = ECONNREFUSED;
      return -1;
   }
   gex_vtest_open_fds[i].session = session;
   return 0;
}

ssize_t gex_vtest_write(int fd, const void *buf, size_t count)
{
   int i = gex_vtest_slot(fd);
   if (i < 0 || gex_vtest_open_fds[i].session < 0) {
      errno = EBADF;
      return -1;
   }
   int n = gex_vtest_js_write(gex_vtest_open_fds[i].session, buf, (int)count);
   if (n < 0) {
      errno = EIO;
      return -1;
   }
   return n;
}

ssize_t gex_vtest_read(int fd, void *buf, size_t count)
{
   int i = gex_vtest_slot(fd);
   if (i < 0 || gex_vtest_open_fds[i].session < 0) {
      errno = EBADF;
      return -1;
   }
   int n = gex_vtest_js_read(gex_vtest_open_fds[i].session, buf, (int)count);
   if (n < 0) {
      errno = EIO;
      return -1;
   }
   /* A reply that is not there is the connection failing, not a wait: the
    * renderer ran the command before this call returned. Reported as an
    * error rather than as end-of-file, because the winsys treats a short
    * read as fatal either way and this says which it was. */
   if (n == 0) {
      errno = EIO;
      return -1;
   }
   return n;
}

int gex_vtest_close(int fd)
{
   int i = gex_vtest_slot(fd);
   if (i >= 0) {
      if (gex_vtest_open_fds[i].session >= 0)
         gex_vtest_js_close(gex_vtest_open_fds[i].session);
      gex_vtest_open_fds[i].fd = -1;
      gex_vtest_open_fds[i].session = -1;
   }
   return close(fd);
}
