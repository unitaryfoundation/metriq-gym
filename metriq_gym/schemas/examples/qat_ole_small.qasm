OPENQASM 3.0;
include "stdgates.inc";
// 6-qubit OLE-style echo circuit for local simulator testing.
// Structure: U (one Trotter step, Ising-like) · V_delta · U†
// Observable: Z_0 ⊗ Z_1 ⊗ Z_2
// Expected OLE value: ~1.0 minus a small delta-squared correction.
qubit[6] q;

// U: one Trotter step
rx(pi/8) q[0];
rx(pi/8) q[1];
rx(pi/8) q[2];
rx(pi/8) q[3];
rx(pi/8) q[4];
rx(pi/8) q[5];
cz q[0], q[1];
cz q[2], q[3];
cz q[4], q[5];
rz(pi/4) q[0];
rz(pi/4) q[1];
rz(pi/4) q[2];
rz(pi/4) q[3];
rz(pi/4) q[4];
rz(pi/4) q[5];
cz q[1], q[2];
cz q[3], q[4];
rz(pi/4) q[1];
rz(pi/4) q[2];
rz(pi/4) q[3];
rz(pi/4) q[4];

// V_delta: small X perturbation, delta = 0.15
rx(0.3) q[0];
rx(0.3) q[2];
rx(0.3) q[4];

// U†: reverse of U
rz(-pi/4) q[3];
rz(-pi/4) q[4];
cz q[3], q[4];
rz(-pi/4) q[1];
rz(-pi/4) q[2];
cz q[1], q[2];
rz(-pi/4) q[5];
rz(-pi/4) q[4];
rz(-pi/4) q[3];
rz(-pi/4) q[2];
rz(-pi/4) q[1];
rz(-pi/4) q[0];
cz q[4], q[5];
cz q[2], q[3];
cz q[0], q[1];
rx(-pi/8) q[5];
rx(-pi/8) q[4];
rx(-pi/8) q[3];
rx(-pi/8) q[2];
rx(-pi/8) q[1];
rx(-pi/8) q[0];
