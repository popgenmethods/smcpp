#include "common.h"

std::mutex mtx;
bool do_progress;
void doProgress(bool x) { do_progress = x; }
