/* SPDX-License-Identifier: MIT */
void wvk_test_publish(void);
__attribute__((constructor)) static void publish(void) { wvk_test_publish(); }
