// quantum teleportation example
// Modified for QuAVL
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
cx q[0],q[1];