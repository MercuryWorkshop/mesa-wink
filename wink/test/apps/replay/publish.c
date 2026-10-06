// SPDX-License-Identifier: MIT
void wink_replay_publish(void);
__attribute__((constructor)) static void publish(void) { wink_replay_publish(); }
