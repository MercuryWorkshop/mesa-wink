/* SPDX-License-Identifier: MIT */
/* The socket the vtest winsys opens, when there is no operating system under
 * it.
 *
 * The winsys talks to virglrenderer over a UNIX socket: write a command, read
 * the reply. In a browser there is no such socket -- and no need for one,
 * because virglrenderer is a wasm module in the same worker as this one. So
 * the five calls the winsys makes are renamed, in these two files only, and
 * answered by handing the bytes straight across.
 *
 * Nothing else changes. The command stream is the one Mesa's virgl driver has
 * always encoded and the one virglrenderer has always decoded; only the pipe
 * between them is shorter. The server answers protocol version 0, so the
 * shared-memory transfer path -- the one that would want a descriptor passed
 * over the socket -- is never negotiated.
 *
 * Force-included into the winsys sources by meson (`-include`), which is why
 * it declares the real calls before renaming them: a later <unistd.h> must
 * declare `read`, not `gex_vtest_read`.
 */
#ifndef GEX_VTEST_OS_H
#define GEX_VTEST_OS_H

#include <stddef.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

int gex_vtest_socket(int domain, int type, int protocol);
int gex_vtest_connect(int fd, const struct sockaddr *addr, socklen_t len);
ssize_t gex_vtest_read(int fd, void *buf, size_t count);
ssize_t gex_vtest_write(int fd, const void *buf, size_t count);
int gex_vtest_close(int fd);

#ifdef GEX_VTEST_OS_RENAME
#define socket  gex_vtest_socket
#define connect gex_vtest_connect
#define read    gex_vtest_read
#define write   gex_vtest_write
#define close   gex_vtest_close
#endif

#endif /* GEX_VTEST_OS_H */
