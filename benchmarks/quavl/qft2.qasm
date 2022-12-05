// quantum Fourier transform
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[3];
crk(2) q[3],q[2];
crk(3) q[3],q[1];
crk(4) q[3],q[0];
h q[2];
crk(2) q[2],q[1];
crk(3) q[2],q[0];
h q[1];
crk(2) q[1],q[0];
h q[0];