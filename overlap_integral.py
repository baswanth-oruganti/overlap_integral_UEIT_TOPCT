#!/usr/bin/env python
# coding: utf-8

# # Wavefunction Overlap Integrals for Quantum Phase Estimation
# 
# This notebook computes wavefunction Overlap Integrals $\langle\Psi_{\rm CAS}|\Psi_{\rm trial}\rangle$ for RHF and PUHF trial wavefunctions against a CASSCF reference. The square of this quantity $|\langle\Psi_{\rm CAS}|\Psi_{\rm trial}\rangle|^2$ is the QPEA success probability: the probability that a single run of the quantum phase estimation algorithm collapses to the desired eigenstate. An example calculation is demonstrated using the antiaromatic TS for electrocyclization of diarylethene-**1** in the paper: 1ts_AA.xyz
# 
# **For full mathematical details see `README.md`.**
# 
# | Section | Content |
# |---|---|
# | 1  | Setup and molecular definition |
# | 2  | RHF and broken-symmetry UHF |
# | 3  | UHF natural orbitals and diradical characters |
# | 4  | CASSCF reference wavefunction |
# | 5  | Diradical Character $y$ and Rotation Angle |
# | 6  | RHF-CASSCF overlap and sanity check |
# | 7  | Spin projection and PUHF-CASSCF overlap |
# | 8  | Rotated PUHF-CASSCF Overlap |
# | 9  | Summary of Key Results |
# | 10 | Additional Info: Comparison with 2-Configurational Wavefunction of Sugisaki et al. |

# ---
# ## Section 1 -- Setup and Molecular Definition
# 

# In[3]:


import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.fci import cistring


# In[18]:


# Molecular definition

# Replace XYZ_FILE with the path to your geometry file.
# The file must be in standard XYZ format:
#   Line 1: number of atoms
#   Line 2: comment (ignored)
#   Lines 3+: Element  X  Y  Z  (Angstrom)
#
# symmetry=False is required: the broken-symmetry UHF must be
# able to break spatial symmetry freely.

XYZ_FILE  = '1ts_AA.xyz'   # <-- replace with your geometry file
BASIS     = 'STO-3G'
N_CAS_ORB = 8    # number of active orbitals
N_CAS_EL  = 8    # number of active electrons

def read_xyz(filename):
    with open(filename) as f:
        lines = f.readlines()
    nat = int(lines[0])
    return ''.join(lines[2:2 + nat])

mol = gto.M(
    atom     = read_xyz(XYZ_FILE),
    basis    = BASIS,
    spin     = 0,
    charge   = 0,
    symmetry = False,
    verbose  = 3
)

print(f'Molecule:     {XYZ_FILE}')
print(f'Electrons:    {mol.nelectron}  ({mol.nelec[0]} alpha, {mol.nelec[1]} beta)')
print(f'Basis:        {BASIS}  ({mol.nao} AOs)')
print(f'Active space: CAS({N_CAS_EL},{N_CAS_ORB})')


# ---
# ## Section 2 -- RHF Reference and Broken-Symmetry UHF
# 
# ### 2.1 Restricted Hartree-Fock
# 
# The RHF wavefunction $|\Phi_{\rm RHF}\rangle$ is a single Slater determinant
# in which electrons of opposite spins occupy identical spatial orbitals.
# For diradicaloid systems, RHF is expected to show low overlap with the exact ground state
# because it cannot represent open-shell singlet character arising from
# near-degenerate frontier orbitals.
# 
# 
# 
# ### 2.2 Broken-Symmetry UHF Initial Guess
# 
# We apply opposite HOMO-LUMO rotations to $\alpha$ and $\beta$ spin-orbital sets.
# Let $\varphi_H, \varphi_L$ be the RHF HOMO and LUMO.  The rotated orbitals are:
# 
# \begin{eqnarray}
# \begin{bmatrix}
# \varphi_H^{\alpha} \\
# \varphi_L^{\alpha} \\
# \end{bmatrix} 
# = \begin{bmatrix} 
# \cos\theta & \sin\theta  \\
# -\sin\theta & \cos\theta  \\
# \end{bmatrix}
# \begin{bmatrix}
# \varphi_H \\
# \varphi_L \\
# \end{bmatrix}
# \end{eqnarray}
# 
# \begin{eqnarray}
# \begin{bmatrix}
# \varphi_H^{\beta} \\
# \varphi_L^{\beta} \\
# \end{bmatrix} 
# = \begin{bmatrix} 
# \cos\theta & -\sin\theta  \\
# \sin\theta & \cos\theta  \\
# \end{bmatrix}
# \begin{bmatrix}
# \varphi_H \\
# \varphi_L \\
# \end{bmatrix}
# \end{eqnarray}
# 
# 
# The **opposite signs** break spin symmetry: $\alpha$ electrons localise preferentially on one lobe, $\beta$ on the other, mimicking the two-determinant open-shell singlet. At $\theta = \pi/4$ the mixing is maximised (50/50 character). The rotated density matrices seed a Newton-DIIS UHF convergence.
# 

# In[19]:


# Step 2a: RHF reference

mf_rhf = scf.RHF(mol).run()
E_rhf  = mf_rhf.e_tot

hono_gap = (mf_rhf.mo_energy[mol.nelectron // 2]
            - mf_rhf.mo_energy[mol.nelectron // 2 - 1]) * 27.211

print(f'E(RHF)        = {E_rhf:.10f} Ha')
print(f'HOMO-LUMO gap = {hono_gap:.3f} eV')


# In[20]:


# Step 2b: Broken-symmetry UHF

theta = np.pi / 4      # 45-degree mixing maximises HOMO-LUMO overlap
C     = mf_rhf.mo_coeff
nocc  = np.count_nonzero(mf_rhf.mo_occ > 0)
hono, luno = nocc - 1, nocc

Ca = C.copy()   # alpha MO coefficients
Cb = C.copy()   # beta  MO coefficients

# Alpha: +theta rotation
Ca[:, hono] =  np.cos(theta)*C[:, hono] + np.sin(theta)*C[:, luno]
Ca[:, luno] = -np.sin(theta)*C[:, hono] + np.cos(theta)*C[:, luno]

# Beta: -theta rotation (opposite sign = spin symmetry breaking)
Cb[:, hono] =  np.cos(theta)*C[:, hono] - np.sin(theta)*C[:, luno]
Cb[:, luno] =  np.sin(theta)*C[:, hono] + np.cos(theta)*C[:, luno]

nalpha, nbeta = mol.nelec
dm_a = Ca[:, :nalpha] @ Ca[:, :nalpha].T
dm_b = Cb[:, :nbeta]  @ Cb[:, :nbeta].T

mf_uhf = scf.UHF(mol).newton()
mf_uhf.kernel(dm0=(dm_a, dm_b))

s2 = mf_uhf.spin_square()[0]
print(f'E(UHF)        = {mf_uhf.e_tot:.10f} Ha')
print(f'<S2>(UHF)     = {s2:.6f}  (ideal singlet: 0.000; open-shell singlet: ~1.0)')
if s2 < 0.1:
    print('  WARNING: <S2> near 0 -- UHF may have collapsed to RHF.')


# ---
# ## Section 3 -- UHF Natural Orbitals and Diradical Characters
# 
# **Natural orbital occupation numbers (NOONs)** are eigenvalues of the spin-summed UHF 1-RDM, obtained by Lowdin transformation (see `README.md` Section 3):
# 
# $$\tilde{\boldsymbol{\gamma}} = \mathbf{S}^{1/2}\,\boldsymbol{\gamma}\,\mathbf{S}^{1/2}, \qquad \mathbf{C}_{\rm NO} = \mathbf{S}^{-1/2}\tilde{\mathbf{C}}$$
# 
# The LUNO occupation $n_{\rm LUNO}$ diagnoses diradical character. **Spin-projected diradical character** (Sugisaki 2019, Eq. 4) corrects for triplet contamination in the raw UHF value:
# 
# $$y_i^{\rm PUHF} = \frac{n_{{\rm LUNO}+i} - 2(1 - n_{{\rm LUNO}+i})}{1 + (1-n_{{\rm LUNO}+i})^2}, \qquad \text{clipped to } [0,1]$$
# 
# $y^{\rm PUHF} = 0$: closed-shell. $y^{\rm PUHF} = 1$: pure diradical. Formula valid only when $n_{\rm LUNO} > 2/3$; clipped to 0 below this threshold.

# ### Step 3a: UHF Natural Orbitals via Lowdin spin projection

# In[54]:


# 3.1  Spin-summed 1-RDM
dm_a_conv, dm_b_conv = mf_uhf.make_rdm1()   # (nao,nao) each
dm_total = dm_a_conv + dm_b_conv

# 3.2  Lowdin square roots
S_ao    = mol.intor('int1e_ovlp')
eigS, U = np.linalg.eigh(S_ao)
S_half     = U @ np.diag(np.sqrt(eigS))       @ U.T   # S^{+1/2}
S_inv_half = U @ np.diag(1.0/np.sqrt(eigS))   @ U.T   # S^{-1/2}

# 3.3  Orthogonalised 1-RDM -> eigenvectors in Lowdin basis
dm_orth          = S_half @ dm_total @ S_half
occ_raw, C_orth  = np.linalg.eigh(dm_orth)

# Sort descending (doubly occupied first)
idx       = np.argsort(occ_raw)[::-1]
occ       = occ_raw[idx]
C_orth    = C_orth[:, idx]

# 3.4  Back-transform to AO basis
C_no = S_inv_half @ C_orth


# Verification: C_no must be S-orthonormal

check = C_no.T @ S_ao @ C_no
print(f'S-orthonormality: max|C_NO^T S C_NO - I| = {np.max(np.abs(check - np.eye(mol.nao))):.2e}')
print('  (should be < 1e-10)')

n_occ = mol.nelectron // 2
print('\nFrontier NOONs:')

print('  ' + '-'*42)
for offset, label in [(-2,'HONO-1'),(-1,'HONO'),(0,'LUNO'),(1,'LUNO+1')]:
    n_i = occ[n_occ + offset]
    char = '<-- diradical' if 0.2 < n_i < 1.8 else ('doubly occ' if n_i > 1.8 else 'virtual')
    print(f'{label:>8}  {n_i:>10.6f}  {char}')

print(f'\nTotal from NOONs: {occ.sum():.6f}  (should = {mol.nelectron})')


# ### Step 3b: Diradical characters calculated using spin-projection formula 
# See Sugisaki, K. *et al.* *ACS Cent. Sci.* **2019**, *5*, 167. 
# 
# Note: These values are given only for comparison purposes. They are not used in our JACS work.

# In[55]:


# Step 3b: Diradical characters


n_pairs = min(n_occ, len(occ) - n_occ)
y_uhf   = np.zeros(n_pairs)
y_puhf  = np.zeros(n_pairs)

for i in range(n_pairs):
    x = occ[n_occ + i]          # n_{LUNO+i}
    y_uhf[i]  = x
    # Sugisaki Eq. 4
    num       = x - 2.0*(1.0-x)
    denom     = 1.0 + (1.0-x)**2
    y_puhf[i] = float(np.clip(num/denom, 0.0, 1.0))

print(f"{'i':>4}  {'n_LUNO+i':>12}  {'y(UHF)':>10}  {'y(PUHF)':>10}  {'Reduction':>12}")
print('  '+'-'*54)
for i in range(min(4, n_pairs)):
    x = occ[n_occ+i]
    print(f'  {i:>2}  {x:>12.6f}  {y_uhf[i]:>10.6f}  {y_puhf[i]:>10.6f}  {y_uhf[i]-y_puhf[i]:>+12.6f}')
print()
y_PUHF = y_puhf[0]


# ---
# ## Section 4 -- CASSCF Reference Wavefunction
# 
# The CASSCF wavefunction over an active space of $n_{\rm el}$ electrons in $n_{\rm orb}$ orbitals:
# 
# $$|\Psi_{\rm CAS}\rangle = \sum_I c_I |D_I\rangle, \qquad \sum_I c_I^2 = 1$$
# 
# $I = (I_\alpha, I_\beta)$ labels alpha/beta occupation strings over the active orbitals. PySCF stores $c_I$ in `mc.ci[ia, ib]`. Orbitals are partitioned into core (doubly occupied in all $|D_I\rangle$), active (variable occupation), and virtual (unoccupied).

# In[23]:


# Step 4: CASSCF reference (reuse mf_rhf from Step 1)

mc = mcscf.CASSCF(mf_rhf, N_CAS_ORB, N_CAS_EL)
mc.fcisolver = fci.direct_spin1.FCI(mol)
mc.kernel()

print(f'E(CASSCF)         = {mc.e_tot:.10f} Ha')
print(f'Correlation energy = {(mc.e_tot - E_rhf)*1000:.4f} mHa')
print(f'Core orbitals:    {mc.ncore}')

# Active-space natural occupations
rdm1_cas    = mc.fcisolver.make_rdm1(mc.ci, mc.ncas, mc.nelecas)
occ_cas, _ = np.linalg.eigh(rdm1_cas)
occ_cas    = np.sort(occ_cas)[::-1]

print('\nCAS active natural occupations:')
for i, o in enumerate(occ_cas):
    char = '<-- diradical' if 0.5 < o < 1.5 else ('doubly occ' if o > 1.5 else 'virtual')
    print(f'  Orb {i:>2}: {o:.6f}  {char}')

# Leading CI configurations
print('\nLeading CI configurations (|c_I|^2 > 0.01):')
na, nb   = mc.nelecas
a_str_all = list(cistring.make_strings(range(mc.ncas), na))
b_str_all = list(cistring.make_strings(range(mc.ncas), nb))
ci_list  = [(mc.ci[ia,ib]**2, mc.ci[ia,ib])
             for ia in range(len(a_str_all))
             for ib in range(len(b_str_all))
             if abs(mc.ci[ia,ib]) > 0.01]
ci_list.sort(reverse=True)
for w,c in ci_list[:5]:
    print(f'  |c|^2 = {w:.5f}   c = {c:+.6f}')


# ## Section 5 — Diradical Character $y$ and Rotation Angle $\Theta$

# ### Section 5a: Diradical character
# 
# $$\boxed{y = n_\mathrm{LUNO} \in [0,1]}$$
# 
# $y$ is twice the weight of the doubly-excited configuration in the perfect-pairing CI scheme, i.e. $y = 2|c_\mathrm{DX}|^2$. This directly gives the $y/2$ factor in the wavefunction coefficients:
# 
# $$|c_\mathrm{DX}|^2 = y/2 \implies c_\mathrm{DX} = \sqrt{y/2}, \quad c_\mathrm{CSS} = \sqrt{1-y/2}$$
# 
# We determine $y = n_\mathrm{LUNO}$ from the **CASSCF** 1-RDM (more accurate than UHF; no spin projection needed). The rotation angle follows from $n_\mathrm{LUNO} = 2\sin^2\Theta$:
# 
# $$\boxed{\Theta = \arcsin\!\sqrt{n_\mathrm{LUNO}/2}}$$
# 
#  $$
#  c_\mathrm{DX} = \sqrt{y/2} = \sqrt{n_\mathrm{LUNO}/2} = \sqrt{\sin^2\Theta} = \sin\Theta
#  $$. 
#  The $y/2$ factor arises naturally from the CASSCF 1-RDM. See README file for the full derivation.

# In[24]:


luno_idx = N_CAS_EL // 2
y        = occ_cas[luno_idx]           # y = n_LUNO directly
n_H      = occ_cas[luno_idx-1]
Theta    = np.arcsin(np.sqrt(y / 2.0)) # from n_LUNO = 2*sin^2(Theta)

c_dx = np.sin(Theta)                   # = sqrt(y/2)
c_cs = np.cos(Theta)                   # = sqrt(1 - y/2)

print(f'y = n_LUNO (CAS) = {y:.6f}  [diradical character, in [0,1]]')
print(f'Theta = {np.degrees(Theta):.4f} deg')
print(f'Verify: 2*sin^2(Theta) = {2*np.sin(Theta)**2:.6f}  (should = n_LUNO)')
print()
print(f'c_CS = cos(Theta) = sqrt(1-y/2) = {c_cs:.6f}')
print(f'c_DX = sin(Theta) = sqrt(y/2)   = {c_dx:.6f}')
print(f'Norm: cos^2 + sin^2 = {c_cs**2 + c_dx**2:.8f}')
print(f'Bloch angle 2*Theta = {np.degrees(2*Theta):.2f} deg  |  '
      f'Fraction to equator Sigma: {np.degrees(Theta)/45:.3f}')


# ---
# ## Section 6 -- RHF-CASSCF Overlap and Sanity Check
# 
# The overlap between $|\Psi_{\rm CAS}\rangle$ and any single Slater determinant $|\Phi\rangle$ is evaluated by the **determinant expansion** :
# 
# $$\langle\Psi_{\rm CAS}|\Phi\rangle = \sum_I c_I\, \det\!\bigl(\mathbf{M}^\alpha[\mathbf{f}^\alpha_I,:]\bigr)\cdot\det\!\bigl(\mathbf{M}^\beta[\mathbf{f}^\beta_I,:]\bigr)$$
# 
# where $\mathbf{M}^\sigma = \mathbf{C}_{\rm CAS}^\top \mathbf{S}_{\rm AO} \mathbf{C}^\sigma_{\rm occ}$ and $\mathbf{f}^\sigma_I$ selects the rows corresponding to occupied CAS MOs for string $I$. The AO metric $\mathbf{S}_{\rm AO}$ is included because the CAS and trial orbital sets need not be mutually orthogonal. See `README.md` for the full derivation.
# 
# **Sanity check:** passing the CASSCF MOs themselves must return $c_0$ (the leading CI coefficient), not 1.0.
# 

# In[25]:


# Core function: CASSCF-determinant overlap 


def compute_overlap(mol, Ca, Cb, mc):
    """
    Compute <Psi_CAS | Phi> where |Phi> has alpha MO coeff Ca
    and beta MO coeff Cb.

    Method: determinant expansion.
    """
    S_ao  = mol.intor('int1e_ovlp')
    C_cas = mc.mo_coeff
    ncore          = mc.ncore
    ncas           = mc.ncas
    neleca, nelecb = mc.nelecas
    nocc_a = ncore + neleca   # total alpha occupied
    nocc_b = ncore + nelecb   # total beta  occupied

    # Overlap matrices: rows=CAS MOs, cols=trial occupied MOs
    M_a = C_cas.T @ S_ao @ Ca[:, :nocc_a]   # (nmo_cas, nocc_a)
    M_b = C_cas.T @ S_ao @ Cb[:, :nocc_b]   # (nmo_cas, nocc_b)

    a_strings = cistring.make_strings(range(ncas), neleca)
    b_strings = cistring.make_strings(range(ncas), nelecb)
    core_idx  = list(range(ncore))
    total     = 0.0

    for ia, a_str in enumerate(a_strings):
        # bit k=1 in a_str -> active orbital k is occupied
        a_idx      = [i for i in range(ncas) if (a_str & (1 << i))]
        full_a     = core_idx + [i + ncore for i in a_idx]  # CAS MO indices
        sub_a      = M_a[full_a, :]   # (nocc_a, nocc_a) square

        for ib, b_str in enumerate(b_strings):
            c_I = mc.ci[ia, ib]
            if abs(c_I) < 1e-12:
                continue
            b_idx  = [i for i in range(ncas) if (b_str & (1 << i))]
            full_b = core_idx + [i + ncore for i in b_idx]
            sub_b  = M_b[full_b, :]
            total += c_I * np.linalg.det(sub_a) * np.linalg.det(sub_b)

    return total


# ---------------------------------------------------------------
# Step 6a: RHF-CASSCF overlap
# ---------------------------------------------------------------
O_rhf = compute_overlap(mol, mf_rhf.mo_coeff, mf_rhf.mo_coeff, mc)
print('RHF-CASSCF overlap')
print(f'  |<CAS|RHF>|   = {abs(O_rhf):.8f}')
print(f'  |<CAS|RHF>|^2 = {abs(O_rhf)**2:.8f}  (QPEA success probability)')

# ---------------------------------------------------------------
# Step 6b: Sanity check -- pass CASSCF MOs as trial wavefunction
# ---------------------------------------------------------------
# Expected: result should equal the leading CI coefficient c0
O_sanity = compute_overlap(mol, mc.mo_coeff, mc.mo_coeff, mc)
c0 = mc.ci[0, 0]
print(f'\nSanity check: <CAS | CAS MOs>')
print(f'  Computed overlap = {O_sanity:.8f}')
print(f'  Leading CI coeff c0 = {c0:.8f}')
if abs(O_sanity - c0) < 1e-5:
    print('  => PASS: overlap matches c0 as expected.')
else:
    print(f'  => WARNING: difference = {abs(O_sanity-c0):.2e}  (expected < 1e-5)')


# ---
# ## Section 7 -- Spin Projection and PUHF-CASSCF Overlap
# 
# **Why spin projection is needed.** The BS-UHF wavefunction $|\Phi\rangle$ is not an eigenfunction of $\hat{S}^2$; it contains triplet contamination. Spin projection extracts the $S=0$ component.
# 
# **Projected state** (Lowdin first-order projection):
# 
# $$|\Psi_{\rm PUHF}\rangle = \mathcal{N}\bigl(|\Phi\rangle + |\tilde\Phi\rangle\bigr)$$
# 
# where $|\tilde\Phi\rangle$ swaps $\alpha \leftrightarrow \beta$ MOs. Swapping cancels the OSS spin contamination,  leaving a pure two-configuration singlet.
# 
# **Normalisation constant** (from $\langle\Psi_{\rm PUHF}|\Psi_{\rm PUHF}\rangle = 1$):
# 
# $$\mathcal{N} = \frac{1}{\sqrt{2 + 2 S_{\rm swap}}}, \qquad S_{\rm swap} = \langle\Phi|\tilde\Phi\rangle$$
# 
# **Final PUHF overlap** :
# 
# $$\boxed{\langle\Psi_{\rm CAS}|\Psi_{\rm PUHF}\rangle = \frac{\langle\Psi_{\rm CAS}|\Phi\rangle + \langle\Psi_{\rm CAS}|\tilde\Phi\rangle}{\sqrt{2 + 2 S_{\rm swap}}}}$$
# 
# QPEA success probability: $P_{\rm QPEA} = |\langle\Psi_{\rm CAS}|\Psi_{\rm PUHF}\rangle|^2$. See `README.md` |for the derivation of $\mathcal{N}$ and the two-determinant approximation note.
# 

# In[34]:


# Step 6a: Raw UHF-CASSCF overlaps (numerator terms for PUHF)

Ca_uhf, Cb_uhf = mf_uhf.mo_coeff

O_phi       = compute_overlap(mol, Ca_uhf, Cb_uhf, mc)  # original
O_phi_tilde = compute_overlap(mol, Cb_uhf, Ca_uhf, mc)  # alpha<->beta swapped

print('Raw UHF overlaps (before spin projection):')
print(f'  <CAS|Phi>       = {O_phi:+.8f}  (alpha=Ca, beta=Cb)')
print(f'  <CAS|Phi_tilde> = {O_phi_tilde:+.8f}  (alpha=Cb, beta=Ca)')

if (abs(O_phi) > 1e-6 and abs(O_phi_tilde) > 1e-6
        and np.sign(O_phi) != np.sign(O_phi_tilde)):
    print('  WARNING: opposite signs -- UHF may not be a broken-symmetry singlet.')
else:
    print('  Sign check: PASS (both overlaps have the same sign).')


# In[35]:


# Step 6b: Swap overlap and PUHF normalisation


def det_overlap_two_dets(mol, Ca1, Cb1, Ca2, Cb2):
    """
    <Phi_1 | Phi_2> = det(S_alpha) * det(S_beta)
    S_alpha = Ca1_occ^T @ S_AO @ Ca2_occ
    S_beta  = Cb1_occ^T @ S_AO @ Cb2_occ
    """
    S_ao          = mol.intor('int1e_ovlp')
    nalpha, nbeta = mol.nelec
    S_a = Ca1[:, :nalpha].T @ S_ao @ Ca2[:, :nalpha]
    S_b = Cb1[:, :nbeta].T  @ S_ao @ Cb2[:, :nbeta]
    return np.linalg.det(S_a) * np.linalg.det(S_b)


# S_swap = <Phi | Phi_tilde>  (Ca2=Cb, Cb2=Ca: the swap)
S_swap = det_overlap_two_dets(mol, Ca_uhf, Cb_uhf, Cb_uhf, Ca_uhf)
S_swap = float(np.clip(np.real(S_swap), -1.0, 1.0))
norm   = np.sqrt(2.0 + 2.0*S_swap)

print(f'S_swap = <Phi|Phi_tilde>     = {S_swap:.8f}')
print(f'Norm = sqrt(2 + 2*S_swap)    = {norm:.8f}')
print(f'Normalisation constant N      = {1.0/norm:.8f}')
if norm < 1e-12:
    print('  WARNING: norm < 1e-12 -- PUHF state ill-defined.')
    print('  The UHF may have collapsed to RHF (S_swap ~ -1).')


# In[36]:


# Step 6c: PUHF-CASSCF overlap 
if norm < 1e-12:
    O_puhf = 0.0
    print('PUHF overlap = 0 (norm ill-defined; see Step 8 warning).')
else:
    O_puhf = abs((O_phi + O_phi_tilde) / norm)

    
    print(f'<CAS|Phi>          = {O_phi:+.8f}')
    print(f'<CAS|Phi_tilde>    = {O_phi_tilde:+.8f}')
    print(f'Sum (numerator)    = {O_phi+O_phi_tilde:+.8f}')
    print(f'Norm (denominator) = {norm:.8f}')
    print('PUHF-CASSCF overlap:')
    print(f'|<CAS|PUHF>|       = {O_puhf:.8f}')
    print(f'|<CAS|PUHF>|^2     = {O_puhf**2:.8f}  (QPEA success probability)')
    prob = O_puhf**2
    label = 'EFFICIENT (>=90%)' if prob>=0.9 else ('MODERATE (50-90%)' if prob>=0.5 else 'INEFFICIENT (<50%)')
    print(f'  QPEA efficiency:    {label}')


# ## Section 8 — Rotated PUHF–CASSCF Overlap
# 
# The RHF MO coefficients are rotated by the analytic angle
# $\Theta = \arcsin\sqrt{n_L/2}$ and Löwdin spin projection is applied.
# 
# The unprojected rotated determinant has $\langle\hat{S}^2\rangle = \sin^2(2\Theta) \neq 0$,
# confirming it contains OSS spin contamination. After Löwdin projection
# the OSS term cancels exactly (it is odd in $\Theta$ and changes sign
# under $\alpha\leftrightarrow\beta$), leaving a pure two-configuration
# singlet:
# 
# $$|\Psi_\mathrm{PUHF}\rangle = \frac{\cos^2\Theta}{\sqrt{\cos^4\Theta+\sin^4\Theta}}
# |\Psi_\mathrm{CSS}\rangle - \frac{\sin^2\Theta}{\sqrt{\cos^4\Theta+\sin^4\Theta}}
# |\Psi_\mathrm{DX}\rangle, \qquad \langle\hat{S}^2\rangle = 0.$$
# 
# Both $|\Psi_\mathrm{CSS}\rangle$ and $|\Psi_\mathrm{DX}\rangle$ are closed-shell
# singlets, so $\langle\hat{S}^2\rangle = 0$ exactly after projection.

# In[37]:


def rotated_determinant(mf,t):
    C=mf.mo_coeff.copy(); n=np.count_nonzero(mf.mo_occ>0); h,l=n-1,n
    Ca=C.copy(); Cb=C.copy()
    Ca[:,h]=np.cos(t)*C[:,h]+np.sin(t)*C[:,l]
    Ca[:,l]=-np.sin(t)*C[:,h]+np.cos(t)*C[:,l]
    Cb[:,h]=np.cos(t)*C[:,h]-np.sin(t)*C[:,l]
    Cb[:,l]=np.sin(t)*C[:,h]+np.cos(t)*C[:,l]
    return Ca,Cb



def rotated_puhf_overlap(mol,Ca,Cb,mc):
    O1=compute_overlap(mol,Ca,Cb,mc); O2=compute_overlap(mol,Cb,Ca,mc)
    ss=float(np.clip(np.real(det_overlap_two_dets(mol,Ca,Cb,Cb,Ca)),-1.,1.))
    norm=np.sqrt(2.+2.*ss)
    return 0. if norm<1e-12 else abs((O1+O2)/norm)

Ca_r,Cb_r=rotated_determinant(mf_rhf,Theta)
S_ao=mol.intor('int1e_ovlp'); na,nb=mol.nelec
Pa=Ca_r[:,:na]@Ca_r[:,:na].T; Pb=Cb_r[:,:nb]@Cb_r[:,:nb].T
s2=na-np.trace(Pa@S_ao@Pb@S_ao)
O_puhf_rot=rotated_puhf_overlap(mol,Ca_r,Cb_r,mc)
print(f'Theta = {np.degrees(Theta):.4f} deg  (analytic, from CASSCF n_LUNO)')
print(f'<S^2> of rotated det = {s2:.4f}  (sin^2(2t)={np.sin(2*Theta)**2:.4f})')
print(f'|<CAS|PUHF>| = {O_puhf_rot:.8f}  P_QPEA = {O_puhf_rot**2:.8f}')


# In[101]:


# <S^2> of the PUHF STATE after Lowdin projection
# After projection:
#   |PUHF> = c_CSS|Psi_CSS> + c_DX|Psi_DX>
# where c_CSS = cos^2(Theta)/N,  c_DX = sin^2(Theta)/N,
#       N = sqrt(cos^4(Theta) + sin^4(Theta))
#
# <S^2>(PUHF) = c_CSS^2 * <Psi_CSS|S^2|Psi_CSS>
#             + c_DX^2  * <Psi_DX |S^2|Psi_DX >


def s2_single_det(mol, Ca, Cb):
    """
    <S^2> for a single Slater determinant.
    <S^2> = (na-nb)^2/4 + (na+nb)/2 - Tr[Pa @ S @ Pb @ S]
    """
    S_ao  = mol.intor('int1e_ovlp')
    na, nb = mol.nelec
    Pa = Ca[:, :na] @ Ca[:, :na].T
    Pb = Cb[:, :nb] @ Cb[:, :nb].T
    return na - np.trace(Pa @ S_ao @ Pb @ S_ao)


# In[105]:



c_t = np.cos(Theta)
s_t = np.sin(Theta)
N_puhf = np.sqrt(c_t**4 + s_t**4)
c_css  = c_t**2 / N_puhf
c_dx   = s_t**2 / N_puhf

# determinants
Ca_css = mf_rhf.mo_coeff.copy()
Cb_css = mf_rhf.mo_coeff.copy()

Ca_dx  = mf_rhf.mo_coeff.copy()
Cb_dx  = mf_rhf.mo_coeff.copy()
Ca_dx[:, hono] = mf_rhf.mo_coeff[:, luno_idx]
Cb_dx[:, hono] = mf_rhf.mo_coeff[:, luno_idx]

s2_css = s2_single_det(mol, Ca_css, Cb_css)
s2_dx  = s2_single_det(mol, Ca_dx,  Cb_dx)


s2_puhf = c_css**2 * s2_css + c_dx**2 * s2_dx 

print('After Lowdin projection:')
print(f'  PUHF coefficients:  c_CSS = {c_css:.8f},  c_DX = {c_dx:.8f}')
print(f'  Ratio c_CSS/c_DX = {c_css/c_dx:.6f}  (cot^2(Theta) = {(c_t/s_t)**2:.6f})')
print(f'  Norm check: c_CSS^2 + c_DX^2 = {c_css**2 + c_dx**2:.10f}')
print(f'  <Psi_CSS|S^2|Psi_CSS> = {s2_css:.10f}')
print(f'  <Psi_DX |S^2|Psi_DX > = {s2_dx:.10f}')
print(f'  <S^2>(PUHF) = {c_css**2:.6f}*{s2_css:.6f}')
print(f'              + {c_dx**2:.6f}*{s2_dx:.6f}')

print(f'             = {s2_puhf:.10f}')
print()
if abs(s2_puhf) < 1e-8:
    print('  => PASS: <S^2>(PUHF) = 0 exactly. Pure singlet confirmed.')
else:
    print(f'  => WARNING: <S^2>(PUHF) = {s2_puhf:.2e} (expected 0)')

# ── PUHF-CASSCF overlap ───────────────────────────────────────────────────
O_puhf_rot = rotated_puhf_overlap(mol, Ca_r, Cb_r, mc)
print()
print(f'|<CAS|PUHF(Theta)>|   = {O_puhf_rot:.8f}')
print(f'P_QPEA                = {O_puhf_rot**2:.8f}')


# ---
# ## Section 9 -- Summary of Key Results

# In[38]:


print('='*68)
print('PUHF OVERLAP -- COMPLETE RESULTS')
print('='*68)
print(f'Molecule:       {XYZ_FILE}')
print(f'Basis:          {BASIS}')
print(f'Active space:   CAS({N_CAS_EL},{N_CAS_ORB})')

print('\n--- Energies ---')
print(f'  E(RHF)   = {E_rhf:.10f} Ha')
print(f'  E(UHF)   = {mf_uhf.e_tot:.10f} Ha')
print(f'  E(CAS)   = {mc.e_tot:.10f} Ha')

s2_val = mf_uhf.spin_square()[0]
print('\n--- UHF diagnostics ---')
print(f'  <S2>(UHF)        = {s2_val:.6f}  (ideal singlet: 0.000)')
print(f'  n_HONO           = {occ[n_occ-1]:.6f}  (ideal doubly-occ: 2.000)')
print(f'  n_LUNO           = {occ[n_occ]:.6f}  (ideal virtual: 0.000)')
print(f'  y0(UHF)          = {y_uhf[0]:.6f}  (diradical char., no spin proj.)')
print(f'y = n_LUNO (CAS)   = {y:.6f}  [diradical character from CAS LUNO, in [0,1]]')


print('\n--- Wavefunction overlaps ---')
print(f"  {'Wavefunction':>12}  {'|overlap|':>12}  {'|overlap|^2':>14}")
print('  '+'-'*42)
print(f"  {'RHF':>12}  {abs(O_rhf):>12.8f}  {abs(O_rhf)**2:>14.8f}")
print(f"  {'PUHF':>12}  {O_puhf:>12.8f}  {O_puhf**2:>14.8f}")
if abs(O_rhf) > 1e-8:
    r = O_puhf / abs(O_rhf)
    print(f'\n  PUHF/RHF improvement: {r:.4f}x (overlap),  {r**2:.4f}x (probability)')
print('\n  S_swap =', f'{S_swap:.8f}',  '  N =', f'{1.0/norm:.8f}')
print('\n'+'='*68)



print('Rotated PUHF OVERLAP')
print('='*68)

print(f'Hilbert Space Mixing Angle, Theta = {np.degrees(Theta):.4f} deg  (analytic, from CASSCF n_LUNO)')
print(f'|<CAS|PUHF>| = {O_puhf_rot:.8f}  Probability_QPEA = {O_puhf_rot**2:.8f}')


# ## 10. Additional Info: Comparison with 2-Configurational Wavefunction of Sugisaki et al.
# These results are given here for comparison purposes only, not included in the submitted JACS paper.
# Two-configurational wavefunction as defined in Sugisaki et al., *ACS Cent. Sci.* **2019**, *5*, 167.
# The Hilbert space mixing angle $\Theta$ is used to calculate weights of |CSS> and |DX> states. 

# In[82]:


def hybrid_overlap(mol, mf_rhf, Theta, y_PUHF, mc):
    """
    |Hybrid> = sqrt(1-y/2)|CS> - sqrt(y/2)|DX>
    where y = n_LUNO = 2*sin^2(Theta), determined from CASSCF 1-RDM.

    Coefficients: c_CS = cos(Theta) = sqrt(1-y/2)
                  c_DX = sin(Theta) = sqrt(y/2)

    The y/2 factor is intrinsic to Sugisaki's definition y = 2*|c_DX|^2.
    It arises naturally from the CASSCF 1-RDM: n_LUNO = 2*sin^2(Theta).

    |CS>: RHF closed-shell determinant
    |DX>: doubly-excited -- HOMO->LUMO in BOTH alpha and beta spin sets
    """
    C    = mf_rhf.mo_coeff.copy()
    nocc = np.count_nonzero(mf_rhf.mo_occ > 0)
    hono, luno = nocc-1, nocc

    Ca_cs=C.copy(); Cb_cs=C.copy()      # |CS>: closed-shell

    Ca_dx=C.copy(); Cb_dx=C.copy()      # |DX>: HOMO->LUMO both spins
    Ca_dx[:,hono]=C[:,luno]
    Cb_dx[:,hono]=C[:,luno]

    c_cs = np.cos(Theta)                # = sqrt(1 - y/2) = sqrt(1 - n_LUNO/2)
    c_dx = np.sin(Theta)                # = sqrt(y/2)     = sqrt(n_LUNO/2)
    
    c_cs_puhf = np.sqrt(1-y_PUHF/2)      # = sqrt(1 - y/2) 
    c_dx_puhf = np.sqrt(y_PUHF/2)        # = sqrt(y/2)

    O_cs = compute_overlap(mol, Ca_cs, Cb_cs, mc) # from CAS NOs
    O_dx = compute_overlap(mol, Ca_dx, Cb_dx, mc) # from CAS NOs
    
    

    return abs(c_cs * O_cs - c_dx * O_dx), abs(c_cs_puhf * O_cs - c_dx_puhf * O_dx)


# In[86]:


O_hybrid_cas, O_hybrid_puhf = hybrid_overlap(mol, mf_rhf, Theta, y_PUHF, mc)

y_val = 2.0*np.sin(Theta)**2
print(f'y = n_LUNO = 2*sin^2(Theta) = {y_val:.6f}')
print(f'y from spin-projected UHF formula = {y_PUHF:.6f}')
print(f'c_CS = cos(Theta) = sqrt(1-y/2) = {np.cos(Theta):.6f}')
print(f'c_DX = sin(Theta) = sqrt(y/2)   = {np.sin(Theta):.6f}')
print(f'Norm: cos^2+sin^2 = {np.cos(Theta)**2+np.sin(Theta)**2:.8f}')
print(f'|<CAS|Hybrid>| (y from CAS NOs)   = {O_hybrid_cas:.8f};  P_QPEA = {O_hybrid_cas**2:.8f}')
print(f'|<CAS|Hybrid>| (y from PUHF formula)   = {O_hybrid_puhf:.8f};  P_QPEA = {O_hybrid_puhf**2:.8f}')


# #### All Overlap Integral Results using Five Different Methods

# In[106]:


import pandas as pd

overlap_df = pd.DataFrame()
overlap_df["Trial Wavefunction"]=["RHF", "PUHF", r"PUHF ($\Theta$)", 
                                  "2-Configuration (y from CAS NOs)",
                                  "2-Configuration (y from PUHF)"]
overlap_df[r"overlap"]=np.array([abs(O_rhf),abs(O_puhf), abs(O_puhf_rot), 
                                                                   abs(O_hybrid_cas),abs(O_hybrid_puhf)])
print(overlap_df)


# ## References
# 1. Sugisaki, K. *et al.* *ACS Cent. Sci.* **2019**, *5*, 167-175.
