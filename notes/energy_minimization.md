% Raw LaTeX source, despite the requested .md filename.
% Compile from the repository root; see the command at the end.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=25mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,microtype,listings,tabularx}
\lstset{basicstyle=\ttfamily\footnotesize,columns=fullflexible,
  keepspaces=true,breaklines=true,showstringspaces=false,
  frame=single,xleftmargin=2mm,xrightmargin=2mm}
\usepackage[hidelinks]{hyperref}
\newcommand{\tr}{\operatorname{Tr}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\stat}{\operatorname{stat}}
\newcommand{\dd}{\mathrm{d}}
\newcommand{\loc}{\mathrm{loc}}
\newcommand{\emb}{\mathrm{emb}}
\newcommand{\qp}{\mathrm{qp}}
\newcommand{\re}{\operatorname{Re}}
\allowdisplaybreaks
\title{Embedding-based energy minimization for RISB}
\author{}
\date{31 October 2026}
\begin{document}
\maketitle

\begin{abstract}
At zero temperature, the implemented RISB constraints give a constrained
Gutzwiller energy objective parameterized by impurity Hamiltonian fields.
At finite temperature, the functional of \texttt{ThermalLatticeSolver} gives a
reduced grand potential in the same fields after eliminating quadratic
lattice/reference variables. Its local curvature, rather than its name,
determines whether minimization is appropriate. Both constructions retain an
interacting impurity solve and avoid explicit slave-boson amplitudes.
Local numerical checks succeed for a half-filled paramagnetic Hubbard model
with $B=1$ and $B=3$, and for two unequal, lattice-coupled impurities at $B=1$. These checks do not establish a general replacement for
root finding, a global thermal variational bound, or superior robustness.
\end{abstract}

\section{Scope and notation}
The starting points are the functionals in \path{src/risb/helpers.py}:
\begin{itemize}
 \item \texttt{get\_lagrangian};
 \item \texttt{get\_thermal\_grand\_potential}.
\end{itemize}
The finite-temperature construction uses \texttt{ThermalLatticeSolver},
not a thermally smeared
\texttt{LatticeSolver} or an impurity thermal average restricted to the
zero-temperature embedding sector.

Use $k_{\rm B}=1$. Fix the physical Hamiltonian, symmetries, embedding sector
where applicable, and gauge. All independent complex matrix entries are
represented by real coordinates. Cluster indices are $i$ and local block
names are \texttt{bl}. Array names below follow the repository:
\begin{center}
\small
\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X@{}}
\toprule
Array & Meaning\\
\midrule
\path{rho_f[i][bl]} & Impurity bath-hole density; the target lattice
quasiparticle density in the zero-temperature inner solve.\\
\path{rho_cf[i][bl]} & Impurity physical--bath hybridization density.\\
\path{rho_qp[i][bl]} & Local quasiparticle density obtained from the shared lattice.\\
\path{rho_c[i][bl]} & Impurity physical-electron density, used for charge
and double counting.\\
\path{R[i][bl]}, \path{Lambda[i][bl]} & Quasiparticle fields.\\
\path{D[i][bl]}, \path{Lambda_c[i][bl]} & Embedding Hamiltonian fields.\\
\bottomrule
\end{tabularx}
\end{center}
Earlier shorthand for \path{rho_cf} has been removed. The mathematical
optimizer coordinates $x,y$ and operator expectations $m$ are explicitly
defined new coordinates, not alternative names for stored density matrices.
The square-root operation $S(\rho_f)$ below abbreviates the operation
inside \texttt{get\_r}; it is not another density matrix.

The contraction in the quasiparticle constraint is
\[
 \Lambda_i:\rho_{f,i}
 =\sum_{\mathrm{bl}}\re\tr
   \bigl({\Lambda_i[\mathrm{bl}]}^{T}\rho_{f,i}[\mathrm{bl}]\bigr).
\]
The same transpose convention is used in the implemented functional.
The \texttt{Delta\_p} argument of \texttt{get\_lagrangian} denotes its
independent quasiparticle-density variable; it is not an alias for
\path{rho_cf}.

The factor $B$ is the number of auxiliary quasiparticle/bath orbitals per
physical orbital. Thus $B=1$ has no added ghosts; $B=3$ has two added ghosts
per physical orbital. For a one-orbital, two-spin impurity, $B=3$ means two
physical plus six bath modes. Its half-filled ground-state embedding sector
has $B+1=4$ particles, whereas its thermal trace includes all particle numbers.

Fixed-filling formulas below use the required quasiparticle filling, including
the implementation's ghost offset. At fixed chemical potential, use the
corresponding grand-potential formulas and the same offset convention.
At finite temperature the chemical-potential terms are those already included
in the three implemented auxiliary Hamiltonians. For prescribed physical
filling, determine $\mu$ through $-\partial_\mu\Omega=N$ and use
$F=\Omega+\mu N$; $\mu$ is not an additional minimization coordinate.


\subsection{Several impurities share one lattice}
The construction is an embedding construction for every correlated cluster
supported by the current projectors. It is not an independent minimization
for each impurity. Use the repository's lists
\[
 \{D_i,\Lambda_{c,i},R_i,\Lambda_i,
   \rho_{f,i},\rho_{cf,i},\rho_{c,i}\},\quad
 i=1,\ldots,n_{\rm clusters}.
\]
Each cluster has an impurity solve. All clusters enter one common
quasiparticle Hamiltonian through
\texttt{helpers.upfold\_r\_lambda}, using
\path{projectors}, \path{gf_struct_mapping}, and the required ghost
projectors. Both $k$-dependent projectors and orbitals outside all correlated
clusters are included through the existing upfolding conventions.

A change of \path{D[i][bl]} can therefore affect the lattice density and
energy on other clusters. Outer coordinates concatenate all independent
cluster fields; the shared lattice couples their gradients and curvature.
Only the prescribed global filling is fixed. Individual impurity charges
may change unless a physical symmetry or explicit constraint fixes them.

The assumptions on the subspaces remain those of the repository:
orthonormal projector rows and nonoverlapping correlated subspaces.
This does not derive a method for overlapping fragments.
Equivalent clusters can share optimizer coordinates and representative
impurity solves; their energy and gradient contributions must still be
counted for every physical cluster. A \texttt{DummyImpuritySolver} supplies
copied/rotated densities; use its representative for overlap energies.

\section{Zero temperature: impurity fields as variational coordinates}
\subsection{Construct admissible embedding data}
Collect the independent entries of all $(D_i,\Lambda_{c,i})$ into
$y=(y_1,\ldots,y_{n_{\rm clusters}})$. For each impurity independently,
temporarily suppress its cluster index in the following local formulas.
With the operator basis reproducing the implemented couplings, write
\begin{equation}
 H_{\emb}(y)=H_{\loc}+\sum_a y_a A_a.
\end{equation}
Solve its normalized ground state in the allowed embedding sector:
\begin{align}
 H_{\emb}(y)|\Phi_y\rangle&=E_{\emb}(y)|\Phi_y\rangle,\\
 m_a(y)&=\langle\Phi_y|A_a|\Phi_y\rangle,\\
 E_{\loc}(y)&=E_{\emb}(y)-y\cdot m(y).
 \label{eq:local-energy}
\end{align}
No amplitudes of $|\Phi_y\rangle$ are outer optimization coordinates.

Read \path{rho_f[i][bl]} and \path{rho_cf[i][bl]} from that impurity solve.
With the local indices temporarily suppressed, the implemented constraints are
\begin{equation}
 \rho_{\qp}=\rho_f,\qquad
 \rho_{cf}=R^{T} S,\qquad S=\sqrt{\rho_f(1-\rho_f)}.
\end{equation}
For $0<\rho_f<1$, define
\begin{equation}
 {R(y)}^{T}=\rho_{cf}(y){S(\rho_f(y))}^{-1}.
 \label{eq:reconstruct}
\end{equation}
This is \texttt{helpers.get\_r}; it applies to rectangular ghost matrices.
Normalization and the embedding constraints now hold by construction,
provided the shared lattice subsequently realizes every target
\path{rho_f[i][bl]}.

\subsection{Eliminate the lattice density constraint}
Let $\hat T_R$ be the physical kinetic operator renormalized by $R$ and
upfolded with the chosen projectors. At the chosen quasiparticle filling,
define
\begin{equation}
 K(\{R_i,\rho_{f,i}\})=
 \min_{\substack{\Psi_0:\ \rho_{\qp,i}=\rho_{f,i}\\\text{for every }i,\mathrm{bl}}}
 \langle\Psi_0|\hat T_{\{R_i\}}|\Psi_0\rangle.
 \label{eq:kinetic-search}
\end{equation}
The band Hamiltonian with constraint field $\Lambda$ gives the dual expression
\begin{equation}
 K(\{R_i,\rho_{f,i}\})=\sup_{\{\Lambda_i\}}
 \left[E_{\qp}^{N}(\{R_i,\Lambda_i\})
       -\sum_i\Lambda_i:\rho_{f,i}\right].
 \label{eq:kinetic-dual}
\end{equation}
Here $E_{\qp}^{N}$ includes the $\Lambda$ term. Its derivative with respect to
$\Lambda$ is the lattice density, so the inner solve is a density-matching
problem in a concave band-energy functional.

Equation~\eqref{eq:kinetic-dual} is exact for targets represented by a
constrained band ground state, including allowed occupations of degenerate
levels. One must not assume that every impurity-generated target is
representable by a pure Slater determinant on a finite mesh. If the pure
constrained search and its dual differ, the dual describes an ensemble
relaxation. The algorithm must stay on the intended representable branch or
explicitly define that relaxation.

\subsection{Objective and gradient}
Combining the constrained lattice energy with Eq.~\eqref{eq:local-energy}
gives
\begin{equation}
 \boxed{\begin{aligned}
 \mathcal E(y)
 &=K(\{R_i(y_i),\rho_{f,i}(y_i)\})\\
 &\quad+\sum_i\bigl(E_{\emb,i}(y_i)-y_i\cdot m_i(y_i)\bigr).
 \end{aligned}}
 \label{eq:zero-objective}
\end{equation}
For admissible targets, this is the Gutzwiller energy of the corresponding
variational state within the implemented approximation. An outer descent
therefore decreases that energy, not a squared residual.

Hellmann--Feynman and the envelope theorem give
\begin{align}
 \dd E_{\emb}&=m\cdot\dd y,\\
 \dd E_{\loc}&=-y\cdot\dd m,\\
 \dd\mathcal E
 &=\boxed{\langle\dd_R\hat T_{\{R_i\}}\rangle_{\qp}
       -\sum_i\bigl(\Lambda_i:\dd\rho_{f,i}+y_i\cdot\dd m_i\bigr).}
 \label{eq:zero-gradient}
\end{align}
The matrix derivatives in Eq.~\eqref{eq:reconstruct} are
\begin{align}
 \dd R^{T}&=(\dd \rho_{cf})S^{-1}-\rho_{cf}S^{-1}(\dd S)S^{-1},\\
 S\,\dd S+\dd S\,S
 &=\dd\rho_f-\rho_f\,\dd\rho_f-\dd\rho_f\,\rho_f.
\end{align}
Thus only impurity responses, lattice one-body expectation values, and a
Sylvester solve are required. Responses may be computed by perturbation
theory, a response solver, or finite differences of impurity observables.

Equivalently, concatenate the cluster expectations into $m$.
Where $m$ provides local coordinates, write
$\widetilde K(m)=K(R(m),\rho_f(m))$ and $J=\partial m/\partial y$. Then
\begin{equation}
 \nabla_y\mathcal E
   =J^{T}\left(\nabla_m\widetilde K-y\right).
 \label{eq:response-projection}
\end{equation}
The bracket contains the remaining embedding-field stationarity conditions.
A small projected gradient need not imply a small bracket if $J$ has null
or nearly null directions. Check the original RISB stationarity conditions.

\subsection{Algorithm and coverage}
\begin{enumerate}
 \item Choose $y$ in the admissible filling and symmetry domain; solve the impurity.
 \item Obtain $(\rho_f,\rho_{cf},R)$ and solve the constrained band problem for $\Lambda$.
 \item Evaluate Eqs.~\eqref{eq:zero-objective} and~\eqref{eq:zero-gradient}.
 \item Take a feasible descent or trust-region step in $y$, and repeat.
 \item Accept convergence only after checking energy, observables, density
       matching, and the remaining field equations.
\end{enumerate}

This parameterization includes every desired RISB solution whose embedding
state is an obtainable ground state of $H_{\emb}(y)$ in the chosen sector.
It does not establish that this family spans every normalized embedding
state. Equivalence of local stationary conditions additionally requires
adequate, nonredundant response directions. Root finding can also locate
energy saddles; an energy minimizer is not intended to reproduce those.

\section{Finite temperature: reduce the implemented thermal functional}
\subsection{Eliminate quadratic variables}
For the physical thermal theory, use grand-canonical impurity traces.
Let $x=(R,\Lambda)$ and $y=(D,\Lambda_c)$ in real, gauge-fixed coordinates:
\begin{align}
 L_T(x,y)&=\sum_i\Omega_{\emb,i}(y_i)+A_T(x,y),\\
 A_T(x,y)&=\Omega_{\qp}(x)-\sum_i\Omega_{0\emb,i}(x_i,y_i).
\end{align}
Here $x=\{R_i,\Lambda_i\}$ and $y=\{D_i,\Lambda_{c,i}\}$ contain all
clusters. $\Omega_{\qp}$ is one shared lattice term, not a sum of
independent impurity lattice energies.
The quadratic reference Hamiltonian is precisely
\texttt{helpers.get\_h0\_emb}; its additive constants must be retained.

For fixed $y$, select a branch satisfying
\begin{equation}
 A_x(x_*(y),y)=0.
 \label{eq:quadratic-inner}
\end{equation}
Only quadratic Hamiltonians enter this inner solve. It may remain a root
problem; it does not require an interacting impurity solve. Gauge directions
must first be removed, and $A_{xx}$ must be invertible on the remaining
coordinates for a smooth local elimination.

Define
\begin{equation}
 \boxed{G_T(y)=\sum_i\Omega_{\emb,i}(y_i)+A_T(x_*(y),y).}
 \label{eq:thermal-objective}
\end{equation}
The envelope theorem yields
\begin{equation}
 \boxed{
 \frac{\partial G_T}{\partial y_{ia}}
 =\left\langle\frac{\partial H_{\emb,i}}{\partial y_{ia}}\right\rangle_{\emb,i}
 -\left\langle\frac{\partial H_{0\emb,i}}{\partial y_{ia}}\right\rangle_{0\emb,i}.}
 \label{eq:thermal-gradient}
\end{equation}
These are impurity/reference bath-density and hybridization mismatches.
The latter include the $R$ factors from $H_{0\emb}$.
Together, Eqs.~\eqref{eq:quadratic-inner} and~\eqref{eq:thermal-gradient}
give all the thermal stationarity equations. No new correlated-entropy
ansatz has been introduced.

\subsection{Curvature and the minimum principle}
Implicit differentiation gives the exact local reduced Hessian
\begin{equation}
 H_G={\left(\sum_i\Omega_{\emb,i}\right)}_{yy}
       +A_{yy}-A_{yx}A_{xx}^{-1}A_{xy}.
 \label{eq:thermal-hessian}
\end{equation}
There is no general theorem that this matrix is positive.
For example, an auxiliary bath-charge field can retain negative curvature.

If the unwanted negative directions are auxiliary, split $y=(u,v)$ and
eliminate them on a locally maximizing branch:
\begin{align}
 G_v(u,v_*(u))&=0,\qquad G_{vv}<0,\\
 F_T(u)&=G_T(u,v_*(u)),\\
 H_F&=G_{uu}-G_{uv}G_{vv}^{-1}G_{vu}.
 \label{eq:second-reduction}
\end{align}
When $H_F>0$, minimizing $F_T$ is an equivalent local minimum formulation.
This additional inner solve still uses impurity solves, not explicit
slave-boson amplitudes.

A spectral split into all positive and negative directions constructs a
local solver for a regular saddle. It does not by itself identify a stable
physical phase: it can also convert an unstable physical branch into a
local minimization problem. A thermodynamic algorithm must retain physical
order parameters, distinguish their instabilities from auxiliary saddle
directions, and compare the appropriate potentials of converged branches.
These reductions do not prove a global Gibbs variational upper bound.

For a thermal Hamiltonian $H(z)$, the needed response follows from
\begin{equation}
 \partial_{ab}\Omega
 =\langle H_{,ab}\rangle
 -\int_0^\beta
   \langle\delta H_{,a}(\tau)\delta H_{,b}(0)\rangle\,\dd\tau,
\end{equation}
with the real symmetric coordinate pairing understood. Quadratic responses
are inexpensive. Interacting responses can be obtained from the impurity
solver or finite differences. Nonlinear $R$ dependence requires the explicit
$\langle H_{,ab}\rangle$ contribution.



\section{Analytic impurity response and the adjoint gradient}
\label{sec:analytic-response}
\emph{Analytic} means differentiating the finite-dimensional impurity equation,
rather than rerunning a displaced Hamiltonian and dividing by a small
parameter step. A numerical linear solve is still required. The fixed
impurity basis and independent real Hamiltonian coordinates are understood.

\subsection{Direct ground-state response}
For an allowed, nondegenerate embedding ground state, define
\[
 H_{\emb}(y)|0\rangle=E_0|0\rangle,\qquad
 A_a=\frac{\partial H_{\emb}}{\partial y_a}.
\]
The operator $A_a$ is obtained by differentiating the actual hybridization,
bath, or local-potential term, including Hermitian conjugates and symmetry
multiplicities. Differentiation gives
\begin{equation}
 \boxed{\begin{aligned}
 (H_{\emb}-E_0)|\partial_a0\rangle
   &=-(A_a-\langle A_a\rangle)|0\rangle,\\
 \langle0|\partial_a0\rangle&=0.
 \end{aligned}}
 \label{eq:sternheimer}
\end{equation}
This projected linear equation is often called the Sternheimer equation.
For a fixed Hermitian observable $O$,
\begin{equation}
 \frac{\partial\langle O\rangle}{\partial y_a}
 =2\re\langle0|O|\partial_a0\rangle.
 \label{eq:observable-response}
\end{equation}
Choose the physical-density, bath-hole, and hybridization bilinears used by
\texttt{get\_rho\_c}, \texttt{get\_rho\_f}, and \texttt{get\_rho\_cf}.
For a complex matrix entry, apply the formula to its Hermitian real and
imaginary parts. Do not apply the $2\re$ formula to a general
non-Hermitian matrix element without that decomposition.
The response of \path{R} follows algebraically through Eq.~\eqref{eq:reconstruct}.

A direct response implementation solves Eq.~\eqref{eq:sternheimer} once per
independent perturbation and contracts each response state with all required
observables. Exact diagonalization can use the already assembled Hamiltonian
and ground state, with an iterative projected solve for large matrices.
A matrix-product-state impurity solver needs its corresponding
response/correction-vector method; such a routine is not supplied by this note.

\subsection{One weighted observable instead of a full response matrix}
The energy gradient needs a weighted sum of density responses, not every
entry of their Jacobian. For example, if at the current trial
\[
 \dd\mathcal E=3\,\dd\langle O_1\rangle
                     -2\,\dd\langle O_2\rangle,
\]
the required derivative is the response of
$\hat O_{\rm grad}=3O_1-2O_2$.
The weights are held fixed during this first-derivative calculation and
recomputed at the next trial.

Earlier discussion called this operator $\hat B$. This note uses
$\hat O_{\rm grad}$ to avoid confusing an observable with the bath-size
factor $B$. It is not a second energy objective:
one must not minimize $\langle\hat O_{\rm grad}\rangle$ in place of
$\mathcal E$.

After matching the shared lattice density, evaluate the expected embedding
fields using the existing helpers, for each \texttt{[i][bl]}:
\begin{lstlisting}[language=Python]
D_expected = helpers.get_d(rho_qp, lopsided_ke_qp)
Lambda_c_expected = helpers.get_lambda_c(
    rho_qp, R, Lambda, D_expected
)
\end{lstlisting}
For fixed local Hamiltonians, the zero-temperature differential can be
rewritten as
\begin{equation}
 \begin{split}
 \dd\mathcal E
 &=\sum_i 2\re\tr\bigl[
       (D_{{\rm expected},i}-D_i)\,\dd\rho_{cf,i}\bigr]\\
 &\quad+\sum_i\re\tr\bigl[
       (\Lambda_{c,{\rm expected},i}-\Lambda_{c,i})
                                      \,\dd\rho_{f,i}\bigr],
 \end{split}
 \label{eq:field-residual-gradient}
\end{equation}
with the block sums and actual operator index conventions understood.
Thus form a Hermitian operator per impurity using the same Hamiltonian
assembly as the solver:
\begin{equation}
 \begin{split}
 \hat O_{{\rm grad},i}
 &=\hat H_{{\rm hybr},i}
       [D_{{\rm expected},i}-D_i]\\
 &\quad+\hat H_{{\rm bath},i}
       [\Lambda_{c,{\rm expected},i}-\Lambda_{c,i}].
 \end{split}
\end{equation}
The CSC extension adds its physical-density response contribution as well.

Solve one projected equation per impurity:
\begin{equation}
 \boxed{\begin{aligned}
 (H_{\emb}-E_0)|\chi\rangle
   &=-(\hat O_{\rm grad}-\langle\hat O_{\rm grad}\rangle)|0\rangle,\\
 \langle0|\chi\rangle&=0.
 \end{aligned}}
 \label{eq:adjoint}
\end{equation}
Hermiticity of the projected resolvent gives every coordinate derivative:
\begin{equation}
 \boxed{\frac{\partial\mathcal E}{\partial y_a}
       =2\re\langle\chi|A_a|0\rangle.}
 \label{eq:adjoint-gradient}
\end{equation}
No full impurity response matrix is needed.
The shared lattice determines the weights; the interacting response solve
itself remains local to each impurity. Sum derivative contributions for
coordinates shared by equivalent clusters.

\begin{lstlisting}[language=Python]
# After one ground-state solve and one constrained band solve:
D_expected = helpers.get_d(rho_qp, lopsided_ke_qp)
Lambda_c_expected = helpers.get_lambda_c(
    rho_qp, R, Lambda, D_expected
)
O_grad = hybridization_operator(D_expected - D)
O_grad += bath_operator(Lambda_c_expected - Lambda_c)

# NEW projected linear solve, not a displaced impurity solve:
chi = projected_response(H_imp, ground_state, O_grad)

# A[a] is dH_imp / dy[a], with all symmetry factors included.
gradient[a] = 2 * real(vdot(chi, A[a] @ ground_state))
\end{lstlisting}
The operator-building and response functions here are design-level
pseudocode. The actual benchmark uses cached public fockit operator matrices
and a projected positive-definite dense linear solve.

\subsection{Finite-temperature response}
For the actual Gibbs state of \texttt{ThermalLatticeSolver}'s impurity,
the exact response of a fixed observable is
\begin{equation}
 \frac{\partial\langle O\rangle}{\partial y_a}
 =-\int_0^\beta
    \langle\delta A_a(\tau)\,\delta O\rangle\,\dd\tau.
 \label{eq:kubo-response}
\end{equation}
The equal-time covariance is generally insufficient if the operators do
not commute. A spectral implementation uses
\[
 \frac{p_m-p_n}{E_m-E_n}
 \longrightarrow -\beta p_n
 \quad\text{when }E_m=E_n,
\]
where $p_n=e^{-\beta E_n}/Z$.
States with negligible thermal populations can still contribute through
virtual transitions from occupied states. A thermal truncation adequate for
expectation values is not automatically an adequate truncation of a
spectral response sum.

The existing reduced thermal first gradient,
Eq.~\eqref{eq:thermal-gradient}, already uses expectation values only.
It does not require a Kubo solve for each evaluation. Thermal responses
are useful for Hessians, Newton steps, and additional constrained fits.

A separate $B=3$ impurity derivative check compared the projected
ground-state response and the spectral thermal response with finite
differences of public impurity expectations. Maximum differences were
$1.1\times10^{-10}$ at $T=0$ and $2.1\times10^{-11}$ at $T=0.5$.
Those earlier checks concerned individual observable derivatives;
the full weighted-adjoint minimization is checked separately below.

\subsection{Using the block2 impurity solvers}
\label{sec:block2-gradient}
The construction does not require exact diagonalization. The applicable
backends, within their supported Hamiltonians and symmetries, are:
\begin{itemize}
 \item Zero temperature: \texttt{Block2ImpuritySolver}.
 \item Finite temperature: \texttt{Block2ThermalImpuritySolver}.
\end{itemize} This is a proposed integration;
no block2 outer minimization or response benchmark has been performed.

At zero temperature, first obtain the ground-state MPS and the existing
\path{rho_f}, \path{rho_cf}, \path{rho_c}, and local energy expectations.
Construct the weighted one-body operator $O_{\rm grad}$ as above.
Equation~\eqref{eq:adjoint} then becomes one projected MPS linear solve per
impurity, followed by transition expectations
$2\re\langle\chi|A_a|0\rangle$ for all field coordinates.
Block2's documented
\href{https://block2.readthedocs.io/en/latest/api/pyblock2.html\#pyblock2.driver.core.DMRGDriver.multiply}{\texttt{DMRGDriver.multiply}}
can solve a linear equation using \texttt{left\_mpo} and supports
\texttt{proj\_mpss}. These are candidate building blocks for the shifted
operator $H_{\emb}-E_0$, the centered right-hand side, and removal of the
ground-state component. Our impurity wrapper does not currently expose
this response operation; an implemented projection and its residual must
be checked rather than assuming a particular call already computes the
required derivative. No finite broadening belongs in the exact static
gradient formula. A regularized inverse gives an approximate response.

The equation above is exact for an exact, isolated eigenstate. A finite
bond-dimension MPS generally has a nonzero full eigenstate residual.
For the exact derivative of the optimized finite-dimensional MPS ansatz,
differentiate its constrained variational stationarity equations, including
the MPS tangent-space and gauge constraints. Alternatively, a correction
vector converged with bond dimension can approximate the exact-state
response. Verify the gradient against derivatives of the objective actually
returned by the solver. A small DMRG energy change alone does not establish
accurate derivatives or selection of the lowest state.

The block2 ground-state solver can average observables over a spin multiplet.
Its response must follow that same state/ensemble prescription. The
nondegenerate one-vector formula cannot silently replace a multiplet
response; see Section~\ref{sec:degenerate-response}.

At finite temperature, the existing block2 solver supplies the densities,
\path{grand_potential}, and \path{entropy} needed by the objective.
Equation~\eqref{eq:thermal-gradient} uses those expectation values directly;
no interacting response solve is needed for this first derivative.
Purification errors, bond truncation, and incomplete state/sector windows
must be small enough that the free energy and gradient are consistent.
In the lowest-state method, changing the included states can introduce
small discontinuities. A window adequate for densities need not be adequate
for resolving the small objective differences used in a line search.

\paragraph{Why the thermal case still has an analytic gradient.}
For a Gibbs state and fixed $T,\mu$,
\[
 \Omega(y)=-T\log\tr e^{-H(y)/T},\qquad
 \partial_a\Omega=\langle\partial_a H\rangle.
\]
After solving $A_x=0$, the implicit derivative of the quadratic inner
solution cancels by stationarity, leaving the impurity/reference
expectation mismatch in Eq.~\eqref{eq:thermal-gradient}.
This is an analytic gradient; it avoids an additional response solve.
The zero-temperature prototype instead evaluates physical energy after
reconstructing $R$ and matching the lattice density. Its differential
contains derivatives of impurity observables, hence the adjoint solve.
The distinction comes from the chosen functional and eliminated variables;
it is not a rule that zero-temperature energy derivatives always require
response, or that thermal observable responses are unnecessary for Hessians.

\section{What changes for a degenerate ground state}
\label{sec:degenerate-response}
The nondegeneracy condition is required by Eq.~\eqref{eq:sternheimer};
it is not a statement that RISB excludes degenerate states.
With several ground states, subtracting one ground-state projector does not
remove the full null space of $H_{\emb}-E_0$.
The current \texttt{FockitImpuritySolver} options
\path{degenerate="ignore"} and \path{degenerate="warn"} both use one state
from the ground subspace. The warning option detects a close second state;
it does not implement the ensemble response below.

\subsection{An isolated multiplet that stays degenerate}
Suppose an isolated, symmetry-protected ground manifold has constant
dimension $g$ under the allowed perturbations.
Let $P=\sum_{s=1}^g|u_s\rangle\langle u_s|$ and choose the equal mixture
$\rho_0=P/g$.
For each member, solve on the complement of the entire ground subspace:
\begin{align}
 (H_{\emb}-E_0)|\partial_a u_s\rangle
   &=-(1-P)A_a|u_s\rangle,\\
 P|\partial_a u_s\rangle&=0.
\end{align}
Then
\begin{equation}
 \frac{\partial\langle O\rangle}{\partial y_a}
 =\frac{2}{g}\re\sum_{s=1}^g
      \langle u_s|O|\partial_a u_s\rangle.
\end{equation}
The result is independent of the basis chosen within the degenerate
subspace. An adjoint variant uses the weighted operator in the same
projected equations; it generally needs a block of $g$ response vectors,
not one vector.

This is the derivative of a specified equal-mixture prescription.
It does not prove that substituting averaged embedding densities preserves
the original pure-state zero-temperature variational ansatz.
That problem may instead require selecting or optimizing a pure vector
within the ground manifold. If relevant observables differ between its
members, the state or ensemble prescription is part of the calculation,
not something a pseudoinverse can decide.

A small matrix check with two identical coupled two-level systems confirms
this projector response against finite differences to $2.5\times10^{-10}$
and confirms invariance under a rotation of the doublet basis.
This checks the subspace-response identity, not a degenerate RISB
minimization or a Mott-state algorithm.

\subsection{A perturbation that splits the multiplet}
If an allowed perturbation splits the ground manifold, one must first
diagonalize its projected perturbation $P A_a P$ to identify the selected
branches. The lowest energy can have a cusp; a unique ordinary gradient
need not exist. Replacing an inverse by a pseudoinverse does not by itself
choose the appropriate ground state or define the derivative.

For the simple splitting doublet,
\[
 H(t)=\diag(t,-t),\qquad E_0(t)=-|t|,
\]
the two directional derivatives at $t=0$ differ.
One needs an explicit branch/ensemble prescription and possibly a nonsmooth
optimization method or different variational coordinates.
A prescribed symmetry sector can help when it is part of the physical
ansatz; arbitrary sector selection must not hide a lower physical branch.
The metallic minimization benchmark below has a nonzero ground-state gap
and does not exercise these degenerate-boundary cases.

\subsection{Why finite temperature is different}
For any finite impurity Hilbert space and $T>0$, the exact Gibbs density
matrix $e^{-H/T}/Z$ is unique and basis independent even when energy levels
are degenerate. It averages the full relevant degenerate subspace.
For the doublet above,
\[
 \Omega_T(t)=-T\ln[2\cosh(t/T)],
 \qquad \Omega_T''(0)=-1/T.
\]
There is no cusp at positive temperature, but the response grows as
$T\to0$. The spectral divided difference in Eq.~\eqref{eq:kubo-response}
has a finite degenerate-level limit; it does not require the
nondegenerate ground-state formula.

This explains why a ground-state degeneracy does not create the same
impurity-state ambiguity in \texttt{ThermalLatticeSolver}.
Relevant degenerate states and sectors must all be included in its trace.
Gauge null directions and other inner-response singularities can still
remain in the full lattice problem.

\section{Practical meaning of the conditions}
These are conditions on the proposed numerical coordinates and nested solves,
not exclusions from RISB theory. Some difficulties also affect root finding.
The extra requirement here is that an inner problem must be solvable at each
accepted outer trial; a simultaneous root solver only needs all its
equations satisfied at the final solution.

\paragraph{Representable target.}
An impurity supplies \path{rho_f[i][bl]}. The shared lattice must realize all
these targets simultaneously at its prescribed filling. For a spinless,
one-orbital lattice fixed at filling $0.5$, a target local density $0.6$ is
incompatible with that filling. With several clusters, individually
reasonable densities can still violate the common charge or lattice
constraints. Reject such trials or use a feasible parameterization; do not
silently fix each cluster's charge to its DFT value.

\paragraph{Gauge choice.}
A consistent rotation of auxiliary orbitals changes matrices but leaves
physical observables unchanged. For example, a quasiparticle rotation gives
$R'=UR$ and $\Lambda'=U\Lambda U^\dagger$, with the other objects transformed
as required by the functional. An optimizer otherwise sees redundant flat
directions. Gauge fixing chooses a basis, not an additional physical
symmetry. Degenerate auxiliary levels require a consistent basis choice
rather than arbitrary eigenvector jumps.

\paragraph{Regular inner branch.}
Nearby outer fields should have a consistently selectable inner solution
that changes smoothly. Nonunique inner solutions, vanishing response, or
branch switching can invalidate the reduced gradient even when the full
stationarity problem still has a solution. Warm starts help but do not prove
uniqueness; repeated evaluations at the same trial must use the same branch.

\paragraph{Degeneracies and boundaries.}
At zero temperature an impurity ground-state level crossing can change
\path{rho_f} or \path{rho_cf} discontinuously. One needs a specified
degenerate-state prescription, directional derivatives, or an active
boundary treatment. Exact finite-temperature Gibbs averages are smooth
through finite-system level degeneracies, although their response can
remain singular or ill-conditioned.

At eigenvalues $0$ or $1$ of \path{rho_f}, the formal inverse square root
is singular. \texttt{get\_r} already uses the repository's pseudoinverse
treatment; this does not by itself establish a derivative through an
active-set change. A Mott state need not have density eigenvalues at these
edges: even at density $1/2$, $R=0$ can collapse bands and introduce
ground-state degeneracy or an undetermined response.

\paragraph{Physical saddle versus minimum.}
A root solver can find a paramagnetic stationary solution even when allowing
magnetization lowers the energy. An unrestricted physical energy minimizer
should leave it. If paramagnetism is deliberately imposed, the minimizer
can instead find the minimum within that restricted symmetry sector.
This distinction does not imply that every paramagnetic solution is
excluded.

\section{Damping, minimum-change steps, and singular fields}
\label{sec:damping}
Damping is a way to control an update, not a separate physical theory.
The residual-based formula and the energy-based formula below have different
objectives. Neither has been implemented or benchmarked as a new RISB
optimizer in this note.

\subsection{A damped step for matching equations}
Let $z$ collect the independent fields of the solve being stabilized, and
let $r(z)$ be its matching residuals. This $z$ is a new optimizer vector,
not a repository array name. First scale coordinates and residuals so that
relative penalties do not depend on arbitrary units. In these scaled real
coordinates define $J_r=\partial r/\partial z$, distinct from the impurity
response $J$ in Eq.~\eqref{eq:response-projection}. At the current point,
choose
\begin{align}
 p_\alpha
 &=\arg\min_p\left\{\tfrac12\|r+J_r p\|^2
                         +\tfrac{\alpha^2}{2}\|p\|^2\right\},\\
 (J_r^{T}J_r+\alpha^2I)p_\alpha&=-J_r^{T}r,\qquad \alpha>0.
 \label{eq:damped-matching}
\end{align}
Use a stable least-squares/SVD implementation rather than explicitly forming
normal equations when conditioning is poor. If $J_r=U\Sigma V^{T}$,
\[
 p_\alpha=-\sum_j\frac{\sigma_j}{\sigma_j^2+\alpha^2}
                    v_j(u_j^{T}r).
\]
An exactly null direction receives no update. A small singular value does
not cause division by that small value; its amplification is bounded by
$1/(2\alpha)$. This suppresses large changes in poorly determined field
combinations. As $\alpha\to0$, the step approaches the minimum-norm
linear least-squares correction. The suppressed combinations are generally
mixtures of entries of $D,\Lambda_c,R,\Lambda$, not selected diagonal
entries that may be frozen independently.

This is damped Gauss--Newton/Levenberg--Marquardt reasoning applied to root
finding through residual minimization. It minimizes a local model of
$\|r\|^2$, not the RISB physical energy. Nonlinear least squares can stop
at a nonzero residual; successful termination must still be followed by
checks of the original matching equations.

\subsection{The related construction for physical energy minimization}
For an admissible reduced energy or free energy $f(z)$, let $g=\nabla f$
and let $B$ approximate its Hessian. A trust-region step solves
\[
 \min_{\|p\|\leq\Delta}
       \left(g^{T}p+\tfrac12p^{T}Bp\right).
\]
Its optimality conditions include
$(B+\lambda I)p=-g$, $B+\lambda I\succeq0$, and $\lambda\geq0$,
with the usual boundary/complementarity condition. Thus an identity shift
can also control an energy step, but it acts on energy curvature rather
than on $J_r^{T}J_r$. The two constructions are not interchangeable.
For indefinite curvature, the trust-region subproblem must handle negative
curvature; an arbitrary fixed positive shift is not a complete algorithm.

For either objective, compare the actual trial improvement with the local
model. Reject inadequate trials, shrink $\Delta$ or increase regularization,
and reevaluate the actual equations. Keep the last accepted state if an
inner solve fails. As model agreement improves, allow larger steps.
A step penalty centered at the current iterate controls each step; a
permanent penalty added to the physical energy about the initial fields
changes the stationary problem unless it is removed or acts only on exact
redundancies. Damping does not by itself enforce representability, total
filling, a gauge, or the regularity of an eliminated inner branch.

\subsection{What SciPy already provides}
The benchmark used \texttt{scipy.optimize.minimize(method="BFGS")} with a
supplied gradient. This is a quasi-Newton method with a Wolfe line search;
it controls step length but does not implement Eq.~\eqref{eq:damped-matching},
identify null matching directions, or impose feasibility constraints.
SciPy's separate \href{https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.BFGS.html}{\texttt{BFGS}} Hessian-update object has a
\texttt{damp\_update} option; that damps a curvature update, not the
singular values of a matching Jacobian, and was not used in our benchmark.

\href{https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html}{\texttt{scipy.optimize.least\_squares}}
provides Levenberg--Marquardt via \texttt{method="lm"} and trust-region
reflective least squares via \texttt{method="trf"}. The latter supports
bounds; its \texttt{lsmr} solver has a documented regularization option for
rank-deficient Jacobians. These are residual minimizers. SciPy's
\href{https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html}{\texttt{minimize}}
also provides energy trust-region methods, such as \texttt{trust-ncg} and
\texttt{trust-krylov}, that require a Hessian or Hessian-vector products.
None automatically recognizes the RISB-specific meaning of saturated or
isolated orbitals. The benchmark's BFGS line-search settings are documented
in the \href{https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html}{SciPy BFGS reference}.

\subsection{What to do about empty/full and isolated orbitals}
For $n$ an eigenvalue of \path{rho_f} or \path{rho_qp},
\[
 s(n)=\sqrt{n(1-n)},\qquad
 s'(n)=\frac{1-2n}{2\sqrt{n(1-n)}}.
\]
The inverse normalization and this derivative become ill-conditioned near
$0,1$. The repository's exact-edge pseudoinverse treatment does not remove
that nearby conditioning problem or establish smooth derivatives across
its threshold. A physical occupation in \path{rho_c} is a different object;
a full physical orbital is not automatically a removable auxiliary mode.
At finite temperature,
$|\partial n/\partial\epsilon|=\beta n(1-n)$ is tiny for a saturated level,
so fitting its energy from occupation remains poorly conditioned.

For a single orbital $a$, zero rows $R_{a,:}=D_{a,:}=0$ remove its direct
kinetic and impurity hybridization couplings. Complete isolation additionally
requires
\[
 \Lambda_{ab}=\Lambda^c_{ab}=0\quad\text{for all }b\ne a.
\]
Otherwise the paths $a\leftrightarrow b$ through $\Lambda$ or $\Lambda_c$
provide indirect coupling to the remaining problem. For an isolated set
of orbitals, only entries connecting that set to its complement must
vanish; mixing within the set is allowed. The null space of a rectangular
$R$ alone is not sufficient to identify removable ghosts.

An example of an undetermined field is the energy of an isolated, empty
quasiparticle level at zero temperature: its occupation remains zero for
any energy above $\mu$. A minimum-change update keeps its previous field
component when the equations have no sensitivity to that component.
For an isolated auxiliary ghost sector at finite temperature, the matching
free-energy contributions can cancel exactly between
$\Omega_{\qp}+\Omega_{\emb}-\Omega_{0\emb}$, leaving genuinely flat
auxiliary directions. In contrast, a nearly saturated finite-temperature
level is generally weakly determined, not exactly free: damping must not
be used to declare a nonzero matching residual converged.

A null direction of the current Jacobian is only a first-order statement.
A decoupled state may have zero gradient but negative curvature toward
recoupling. Recheck actual residuals and admissibility after each step;
allow directions to reenter as sensitivity changes, and test recoupling
curvature or controlled perturbations before treating a decoupled solution
as stable. No general guarantee or benchmark for these failure cases has
been established here.

\section{Pseudocode using repository names}
The following is design-level pseudocode, not a newly added solver API\@.
\texttt{S} is a configured \texttt{LatticeSolver} or
\texttt{ThermalLatticeSolver} used to hold the problem and trial state.
All stored matrices keep their existing list-of-block-dictionaries layout.
Functions marked \texttt{NEW} are optimizer wrappers still to be implemented.
For clarity these two examples assume fixed local Hamiltonians and no
density-dependent \path{double_counting} hook. The CSC extension below
explains the additional terms.

Do not call the existing \texttt{one\_cycle} inside an objective evaluation:
it changes embedding fields that the objective must hold fixed at their
trial values. Individual band, impurity, and residual operations can be reused.

\subsection{Zero-temperature objective evaluation}
\begin{lstlisting}[language=Python]
def evaluate_energy(y):
    # NEW: unpack independent coordinates for every cluster.
    S.D, S.Lambda_c = unpack_embedding_fields(y)

    # Existing method: set each impurity Hamiltonian, solve it,
    # and store rho_f[i][bl], rho_cf[i][bl], rho_c[i][bl].
    S.solve_impurities()

    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            S.R[i][bl] = helpers.get_r(
                S.rho_cf[i][bl], S.rho_f[i][bl]
            )

    # NEW GLOBAL solve, holding every R[i] and rho_f[i] fixed.
    # Adjust all Lambda[i] together so rho_qp[i] == rho_f[i].
    # Also enforce the prescribed total filling.
    feasible = match_lattice_density(
        S, target_rho_qp=S.rho_f
    )
    if not feasible:
        return infeasible_trial  # Shrink/reject the step.

    # energies_qp contain Lambda, but not a -mu shift here.
    # kweights already contain the integration weights.
    energy_qp = sum(
        np.sum(S.energies_qp[bl] * S.kweights[bl])
        for bl in S.energies_qp
    )
    constraint_energy = sum(
        np.trace(
            S.Lambda[i][bl].T @ S.rho_f[i][bl]
        ).real
        for i in range(S.n_clusters)
        for bl, _ in S.gf_struct[i]
    )
    h_loc_energy = [
        imp.overlap(imp.h_int + imp.h0_loc).real
        for imp in S.impurity_solvers
    ]
    energy = (
        energy_qp - constraint_energy + sum(h_loc_energy)
    )

    # NEW: evaluate the differential derived above.
    # Use finite differences or the analytic adjoint response.
    gradient = embedding_gradient_zero(S, y)
    return energy, gradient

# NEW optimizer layer with feasible steps and residual checks.
y = constrained_minimize(evaluate_energy, initial_fields)
\end{lstlisting}
The overlap line is shown for direct impurity solvers.
For equivalent clusters represented by dummy solvers, obtain the overlap
from the representative and include every cluster's contribution.

\noindent\begin{minipage}{\linewidth}
At convergence check \path{rho_qp[i][bl] - rho_f[i][bl]}, total filling,
and the remaining field equations for every cluster and block:
\begin{lstlisting}[language=Python]
D_expected = helpers.get_d(rho_qp, lopsided_ke_qp)
Lambda_c_expected = helpers.get_lambda_c(
    rho_qp, R, Lambda, D
)
# Check D - D_expected and Lambda_c - Lambda_c_expected.
\end{lstlisting}
\end{minipage}
These arguments denote one \texttt{[i][bl]} entry.
\texttt{get\_f1} and \texttt{get\_f2} alone do not establish convergence
of this minimizer: its construction already satisfies those embedding
constraints when the band density matches.

\subsection{Finite-temperature objective evaluation}
This pseudocode uses a fixed \path{S.mu}. The impurity solvers must have the
same \path{beta} and include all particle numbers and required symmetry sectors.
\begin{lstlisting}[language=Python]
def evaluate_thermal_potential(y):
    # NEW coordinate unpacking; all clusters are included.
    S.D, S.Lambda_c = unpack_embedding_fields(y)

    # NEW GLOBAL quadratic solve for R and Lambda.
    # D and Lambda_c stay fixed. Solve:
    #   rho_qp_lattice == rho_qp_reference
    #   rho_qpf_reference @ D == lopsided_ke_qp.T
    # No interacting impurity solves enter this inner problem.
    solve_quadratic_inner(S)

    # Existing methods: solve all thermal impurities, then
    # calculate impurity/reference and lattice/reference errors.
    S.solve_impurities()
    S.update_residuals()

    grand_potential = helpers.get_thermal_grand_potential(
        energies=S.energies_qp,
        R=S.R,
        Lambda=S.Lambda,
        D=S.D,
        Lambda_c=S.Lambda_c,
        beta=S.beta,
        mu=S.mu,
        impurity_grand_potential=[
            imp.grand_potential
            for imp in S.impurity_solvers
        ],
    )

    # NEW: derivatives in the chosen real coordinates.
    # Use S.f1, S.f2 with the correct factors, conjugations,
    # and sums for coordinates shared by equivalent clusters.
    gradient = thermal_field_gradient(S, y)
    return grand_potential, gradient
\end{lstlisting}
The outer loop checks reduced curvature. If it is positive on the intended
physical coordinates, minimize this objective. If auxiliary negative
directions remain, eliminate those on a selected local maximum before
minimizing the retained directions, as in Eq.~\eqref{eq:second-reduction}.
Do not merely pass a known saddle objective to an ordinary minimizer.

Thermal convergence requires all four matrix matching equations,
physical filling, and observables. The existing
\path{S.bath_fit_error} measures lattice/reference matching;
\path{S.f1} and \path{S.f2} measure impurity/reference matching.
\path{S.self_consistency_error} is assigned by \texttt{solve};
a new optimizer must calculate its own current error rather than reuse a
stale attribute from an earlier solve.

\subsection{Existing building blocks versus new routines}
\begin{center}
\small
\begin{tabularx}{\linewidth}{@{}lX@{}}
\toprule
Operation & Repository components\\
\midrule
All impurity solves &
\path{S.solve_impurities}; internally \texttt{set\_h\_imp},
\texttt{solve}, \texttt{get\_rho\_f}, \texttt{get\_rho\_cf},
and \texttt{get\_rho\_c}.\\
Shared bands &
\path{helpers.upfold_r_lambda}, \path{helpers.get_h_qp},
\path{numpy.linalg.eigh}.\\
Lattice densities and kinetic data &
\path{helpers.get_rho_qp}, \path{helpers.get_ke}.\\
Zero-temperature reconstruction &
\path{helpers.get_r}.\\
Zero-temperature field checks &
\path{helpers.get_d}, \path{helpers.get_lambda_c}.\\
Quadratic reference &
\path{helpers.get_h0_emb_densities},
\path{helpers.get_h0_emb_grand_potential}.\\
Thermal objective and checks &
\path{helpers.get_thermal_grand_potential},
\path{helpers.get_thermal_f1}, \path{helpers.get_thermal_f2},
\path{S.update_residuals}.\\
New optimizer routines &
Coordinate packing, constrained density matching, quadratic inner
stationarity, response gradients, branch-aware feasible steps,
and residual-based acceptance.\\
\bottomrule
\end{tabularx}
\end{center}

The existing \texttt{get\_thermal\_r\_lambda} fits
impurity/reference matching at fixed \path{D} and \path{Lambda_c}.
The existing \texttt{get\_thermal\_d\_lambda\_c} fits
lattice/reference matching at fixed \path{R} and \path{Lambda}.
The proposed quadratic inner solve instead varies \path{R} and
\path{Lambda} to satisfy lattice/reference matching at fixed embedding
fields. It requires a new wrapper; neither existing fit is that operation.

For fixed physical particle number away from a symmetry that fixes filling,
an additional charge constraint is needed. Formally the canonical reduction
is
\[
 G_{T,N}(y)=\stat_{\mu}\bigl[G_T(y,\mu)+\mu N_{\rm target}\bigr].
\]
At a full stationary solution the charge conjugate to $\mu$ must agree with
the physical filling. This is an additional coupled inner problem containing
the impurity grand potential, not just the quadratic calculation.
An implementation must differentiate the consistent full functional and
check physical charge; it must not change $\mu$ and then reuse a
fixed-$\mu$ gradient without accounting for that elimination.
The finite-temperature tests here use particle--hole symmetry and fixed
$\mu=0$, so this general canonical extension has not been run outside those
particle--hole-symmetric examples.

\section{Explicit \texorpdfstring{$B=3$}{B=3} check}
\subsection{Model and coordinates}
Use the repository's 400 equally weighted Bethe-dispersion samples of
half bandwidth one, $U=1$, a paramagnet, and
\[
 H_{\loc}=U n_\uparrow n_\downarrow
          -\frac U2(n_\uparrow+n_\downarrow).
\]
At half filling $\mu=0$. This local particle--hole shift is included in all
reported energies.

For $B=3$, diagonalize the bath/quasiparticle fields and fix real orbital
signs. In the particle--hole symmetric sector, use
\begin{align}
 R&={(r_1,r_0,r_1)}^{T},& \Lambda&=\diag(-\ell,0,\ell),\\
 D&={(d_1,d_0,d_1)}^{T},& \Lambda_c&=\diag(-b,0,b).
\end{align}
Thus $x=(r_0,r_1,\ell)$ and $y=(d_0,d_1,b)$.
This is a restricted, symmetry-preserving benchmark, not a general
multi-orbital parameterization.

At $T=0.2$ the reference solution has approximately
\[
 x=(0.83510887,\,0.38851601,\,0.85071464),\qquad
 y=(0.32227769,\,0.27003717,\,0.87883223).
\]
The reduced Hessian in these $y$ coordinates is
\[
 H_G\simeq
 \begin{pmatrix}
 4.58001164&3.34863607&-0.47151641\\
 3.34863607&3.04299367&-0.29810055\\
 -0.47151641&-0.29810055&0.05256575
 \end{pmatrix}.
\]
Its eigenvalues are approximately
\[
 0.000352741,\quad 0.385529911,\quad 7.289688418.
\]
Therefore $G_T$ itself has a local minimum in this symmetric subspace.
The condition number is about $2.1\times10^4$; even this small example
contains a soft direction. No additional maximization is needed here.
Particle--hole symmetry has already fixed the separate bath-charge field
whose negative curvature appears in an unrestricted $B=1$ calculation.

\subsection{Observables and residuals}
The ground-state checks use a pure embedding ground state. The reference
\texttt{LatticeSolver} uses $\beta=10^6$ smearing, whereas the minimization's
band occupations are sharp at fixed filling. The thermal checks use
\texttt{FockitThermalImpuritySolver} over all particle numbers with
\path{method="lowest-scan"} and relative Boltzmann cutoff $10^{-14}$.

All minimizations start from a $3\%$ perturbation of a known solution in the
stated coordinates. They are local restart checks. Thermal gradients use
Eq.~\eqref{eq:thermal-gradient}; zero-temperature gradients use
Eq.~\eqref{eq:zero-gradient}, with central impurity-response differences
of step $10^{-5}$. BFGS is used for the outer optimization. Its gradient tolerance is
$10^{-8}$ at $T=0$, $10^{-8}$ for thermal $B=1$, and $10^{-9}$ for thermal
$B=3$. The restart DIIS tolerance is $10^{-10}$; reference generation uses
$10^{-11}$. The inner thermal root tolerance is $10^{-10}$.

The reference observables are:
\begin{center}
\small
\begin{tabular}{@{}rrrrr@{}}
\toprule
$T$ & $B$ & $\langle n_\uparrow n_\downarrow\rangle$ & $Z_{\rm qp}$ & $E$\\
\midrule
0 & 1 & 0.1763677731 & 0.9132527225 & -0.7112226716 \\
0 & 3 & 0.1669905342 & 0.7626956053 & -0.7159949344 \\
0.2 & 1 & 0.1481348953 & 0.8641517269 & -0.6430630214 \\
0.2 & 3 & 0.1419626883 & 0.6974068277 & -0.6527934068 \\
\bottomrule
\end{tabular}
\end{center}
Here $Z_{\rm qp}={[1-\partial_\omega\Sigma(0)]}^{-1}$ is calculated with
\texttt{get\_z}. For ghosts it is distinct from the total spectral weight
$R^\dagger R$.

Absolute differences between the minimization and reference solutions are:
\begin{center}
\small
\begin{tabular}{@{}rrrrrr@{}}
\toprule
$T$ & $B$ & $|\delta d|$ & $|\delta Z_{\rm qp}|$
 & $\max|\delta\Sigma|$ & $|\delta E|$\\
\midrule
0 & 1 & $5.55\times10^{-12}$ & $1.31\times10^{-11}$ & $4.70\times10^{-11}$ & $1.11\times10^{-15}$ \\
0 & 3 & $1.21\times10^{-10}$ & $9.54\times10^{-10}$ & $7.86\times10^{-9}$ & $1.09\times10^{-14}$ \\
0.2 & 1 & $3.81\times10^{-10}$ & $7.03\times10^{-9}$ & $5.32\times10^{-8}$ & $4.62\times10^{-10}$ \\
0.2 & 3 & $3.09\times10^{-12}$ & $2.11\times10^{-9}$ & $8.52\times10^{-10}$ & $2.28\times10^{-14}$ \\
\bottomrule
\end{tabular}
\end{center}
At $T=0$, the imaginary-frequency sample is
$\omega=i(0.1,0.3,1,3)$; these are not finite-temperature Matsubara points.
At $T=0.2$, it consists of the first five positive fermionic Matsubara
frequencies. Self-energies, rather than gauge-dependent ghost matrices,
are compared.

Densities agree to $10^{-15}$. Thermal entropies agree within
$2.4\times10^{-9}$ for $B=1$ and $1.2\times10^{-13}$ for $B=3$;
the grand-potential differences are below displayed double precision.
Thermal residual checks include all four equations: lattice/reference
density, lattice/reference kinetic matching, impurity/reference bath
density, and impurity/reference hybridization. Their largest errors are
$2.1\times10^{-9}$ and $1.8\times10^{-11}$ for $B=1,3$, respectively.

At $T=0$, constrained lattice-density errors are below $1.5\times10^{-14}$.
The remaining $D,\Lambda_c$ field equations agree to
$2.9\times10^{-11}$ for $B=1$ and $1.1\times10^{-7}$ for $B=3$.
The latter is substantially larger than the projected outer gradient,
$8.6\times10^{-10}$: Eq.~\eqref{eq:response-projection} explains why
an optimizer's gradient tolerance alone is not a reliable RISB stopping rule.


\subsection{Two inequivalent impurities coupled by the lattice}
A further check uses two separate one-orbital impurities with
$U_1=1$, $U_2=1.4$, $B=1$ on each, and one shared spin-block dispersion
\[
 h_0(k)=
 \begin{pmatrix}
 \epsilon_k&0.2\\
 0.2&\epsilon_k
 \end{pmatrix}.
\]
Use \path{projectors[0][bl]} $=(1,0)$ and
\path{projectors[1][bl]} $=(0,1)$, with the same 400 Bethe samples.
Both clusters are paramagnetic and particle--hole symmetric.
Each local interaction includes its own $-U_i n_i/2$ shift.
There are two impurity solves per outer evaluation and one common lattice
solve; the nonzero off-diagonal hopping makes the energy genuinely coupled.

Starting from a $3\%$ perturbation of the two embedding hybridizations,
the local minimum reproduces the existing two-impurity self-consistent
solution:
\begin{center}
\small
\begin{tabular}{@{}rrrrr@{}}
\toprule
$T$ & $\max_i|\delta d_i|$ & $\max_i|\delta Z_i|$
 & $|\delta E\ \text{or}\ \delta\Omega|$ & Residual\\
\midrule
0 & $2.45\times10^{-9}$ & $5.46\times10^{-9}$
 & $6.66\times10^{-16}$ & $1.53\times10^{-8}$\\
0.2 & $5.64\times10^{-9}$ & $1.11\times10^{-7}$
 & $2.22\times10^{-15}$ & $3.20\times10^{-8}$\\
\bottomrule
\end{tabular}
\end{center}
The reference double occupancies, ordered by cluster, are
\[
 \begin{array}{rl}
 T=0:   &(0.1803115036,\,0.1530367391),\\
 T=0.2: &(0.1545365848,\,0.1186146838).
 \end{array}
\]
Densities agree within $7\times10^{-16}$.
This check uses numerical objective gradients with BFGS tolerance $10^{-7}$.
The residual at $T=0$ includes the remaining hybridization-field equations;
at finite temperature it includes the impurity/reference matching error,
with the quadratic equations solved separately.
This validates a coupled two-impurity example, not a CSC calculation,
arbitrary fillings, or arbitrary multi-orbital projectors.


\subsection{Full minimization with the weighted analytic response}
\label{sec:adjoint-benchmark}
This earlier setup reused reference constraint-field coefficients as the
initial guess of each band-density fit. Section~\ref{sec:matched-diis}
records a new comparison with DIIS using the common perturbed initial state
for those fits. The earlier and newer timings should not be mixed.
The preceding impurity derivative checks did not constitute a complete
minimization with the weighted observable. A new full comparison now does.
Use the same $T=0$, $U=1$, half-filled Bethe problem, 400 samples,
particle--hole/paramagnetic coordinates, reference states, and $3\%$ initial
perturbation as above. Compare:
\begin{enumerate}
 \item The original finite-difference impurity-response gradient.
 \item The weighted-observable adjoint gradient of
       Eqs.~\eqref{eq:adjoint}--\eqref{eq:adjoint-gradient}.
\end{enumerate}
Both use the same constrained band calculation, BFGS, and gradient tolerance
$10^{-8}$. The only change to the outer algorithm is the gradient routine.
The embedding Hamiltonian matrices have dimensions $6$ for $B=1$ and $70$
for $B=3$. Ground-state excitation gaps at the perturbed starting points are
approximately $0.6221$ and $0.2473$, respectively, so these tests use isolated
nondegenerate ground states.

At three parameter points, including the starting fields and the reference
solution, the largest adjoint/finite-difference gradient differences are
$4.5\times10^{-12}$ for $B=1$ and $4.8\times10^{-10}$ for $B=3$.
Full minimizations give:
\begin{center}
\small
\begin{tabular}{@{}rlrrr@{}}
\toprule
$B$ & Gradient & Ground-state solves & Response solves & Median time (ms)\\
\midrule
1 & Finite differences & 18 & 0 & 12.04 \\
1 & Adjoint & 6 & 6 & 4.60 \\
3 & Finite differences & 140 & 0 & 326.85 \\
3 & Adjoint & 20 & 20 & 151.13 \\
\bottomrule
\end{tabular}
\end{center}
Times are medians of five full minimizations, alternating method order.
The common reference/model preparation and one-time response-operator cache
are excluded. Cache construction takes approximately $0.59$ ms for $B=1$
and $1.37$ ms for $B=3$. Seven alternating evaluations at the same perturbed
point give median objective-plus-gradient times:
\[
\begin{array}{crr}
 &\text{finite differences}&\text{adjoint}\\
 B=1&1.953\ {\rm ms}&0.735\ {\rm ms}\\
 B=3&17.644\ {\rm ms}&8.960\ {\rm ms}
\end{array}
\]
Both methods take five BFGS iterations for $B=1$ and eighteen for $B=3$.
There are six and twenty objective evaluations, respectively.
Finite differences use $1+2\dim(y)$ ground-state solves per evaluation:
three at $B=1$ and seven at $B=3$.
The adjoint instead uses one ground-state solve and one projected response
solve per evaluation. It does not need one response solve per parameter.

The median full speedups are about $2.62$ and $2.16$, not the raw factors
three and seven suggested by ground-state solve counts.
The common lattice-density inversion and other work remain in each
evaluation. Cached operator construction and dense projected solves are
suitable for these small spaces; large impurities need a sparse or
matrix-product-state response implementation and separate measurement.

For $B=3$, the adjoint minimum agrees with the root reference to
$3.6\times10^{-10}$ in double occupancy,
$5.7\times10^{-10}$ in quasiparticle weight, and
$1.5\times10^{-9}$ in the sampled self-energy.
Its remaining field-equation error is about $3.6\times10^{-9}$,
versus $1.1\times10^{-7}$ for the finite-difference prototype.
Both energies agree with the reference to approximately $10^{-14}$.
These are complete impurity-based minimizations using the weighted
$\hat O_{\rm grad}$, not only checks of individual response entries.
They are local metallic tests; they do not benchmark a degenerate Mott state.

The comparison is specifically for the zero-temperature response-dependent
gradient. The earlier finite-temperature minimizer already uses an
expectation-value first gradient and does not need this replacement.
These measurements establish an improvement over the finite-difference
prototype in the stated cases, not a general advantage over DIIS or every
impurity solver.

\section{Efficiency, robustness, and limits}
\subsection{Illustrative cost}
The local restart comparison uses the same initial physical matrices
obtained from the perturbed embedding fields and the corresponding inner
lattice solution. Reference generation and diagnostic Hessian construction
are excluded. The existing solver uses its default DIIS driver; the
prototype uses BFGS\@. Timings are single short runs on this machine, with
different code paths, cache histories, and stopping tolerances. They are
not a controlled performance study. This cost table refers to the
single-impurity checks; the additional coupled-impurity check was not timed.

\begin{center}
\small
\begin{tabular}{@{}rrrrrr@{}}
\toprule
$T$ & $B$ & DIIS impurity solves & Minimum solves & DIIS (s) & Minimum (s)\\
\midrule
0 & 1 & 8 & 18 & 0.008 & 0.012 \\
0 & 3 & 9 & 140 & 0.028 & 0.324 \\
0.2 & 1 & 9 & 5 & 0.034 & 0.011 \\
0.2 & 3 & 6 & 24 & 0.105 & 0.439 \\
\bottomrule
\end{tabular}
\end{center}

At $T=0$, a gradient evaluation uses $1+2\dim(y)$ impurity solves with
finite-difference response: three for $B=1$ and seven for $B=3$.
The analytic adjoint comparison in Section~\ref{sec:adjoint-benchmark}
demonstrates a reduction of this cost in the stated cases. At finite
temperature the envelope gradient requires only one interacting impurity
solve per outer evaluation, but each evaluation also solves the quadratic
inner problem.

DIIS is cheaper in the tested $B=3$ examples. The tiny thermal $B=1$
minimum is faster in this particular local check. Neither result establishes
a general ranking. A useful benchmark must match final residuals and
observable tolerances, include cold and continuation starts, and measure
many-body solves, quadratic work, wall time, and failed runs.


\subsection{Rerun against the current DIIS implementation}
\label{sec:matched-diis}
This comparison reruns both temperatures and both bath sizes with the current
default \texttt{DIIS(error\_fun="root")} returned by the lattice solvers.
It uses Python 3.14.8, NumPy 2.5.4, and SciPy 1.18.1, with BLAS/OpenMP
thread limits set to one before import.
The model remains the half-filled paramagnetic Bethe problem:
$U=1$, 400 equally weighted samples, $B=1,3$, and $T=0,0.2$.

Each method starts from the same physical matrices obtained from a
$3\%$ embedding-field perturbation and its constrained inner solution.
Unlike the earlier timing comparison, repeated inner fits start from that
common perturbed initial state, not from the exact reference solution.
Each repeat uses fresh optimizer history. Impurity caches are warmed before
timing. Model/reference construction, initial-state matching, and one-time
operator-cache preparation are excluded.
Methods are run serially in rotating order; the table reports medians of
seven runs, not simultaneous timings.

The outer energy minimizer is SciPy BFGS\@:
\begin{lstlisting}[language=Python]
minimize(
    objective, initial_fields, method="BFGS", jac=True,
    options={"gtol": 1e-8, "maxiter": 150},
)
\end{lstlisting}
At $T=0$, the supplied gradient is either the finite-difference impurity
response or the weighted analytic adjoint response.
At finite temperature, the analytic gradient is the expectation-value
mismatch, without a Sternheimer/Kubo solve.
For the additional finite-temperature numerical-gradient baseline,
use \path{jac="3-point"} on the entire reduced grand potential.
Thus the two numerical-gradient columns use different constructions, which
must not be conflated.

DIIS uses \path{S.solve(tol=1e-10, maxiter=300)} and the existing default
callbacks. Inner constraint fits use \path{scipy.optimize.root} with
tolerance $10^{-10}$. The minimizer uses reduced coordinates that enforce
symmetry and fix gauge; DIIS uses its existing parameterization.
Both solve the same physical restricted problem, but this is a comparison
of complete implementations, not of abstract optimizer steps in identical
coordinates.

\begin{center}
\small
\begin{tabular}{@{}rrrrrrr@{}}
\toprule
$T$ & $B$ & \shortstack{DIIS\\(ms)} & \shortstack{FD\\(ms)}
 & \shortstack{Analytic\\(ms)}
 & \shortstack{FD/\\DIIS} & \shortstack{Analytic/\\DIIS}\\
\midrule
0 & 1 & 8.03 & 11.49 & 4.42 & 1.43 & 0.55 \\
0 & 3 & 27.65 & 372.68 & 860.41$^{*}$ & 13.48 & 31.12$^{*}$ \\
0.2 & 1 & 31.92 & 32.44 & 10.88 & 1.02 & 0.34 \\
0.2 & 3 & 96.67 & 4210.06 & 600.93 & 43.55 & 6.22 \\
\bottomrule
\end{tabular}
\end{center}
A ratio below one means the minimizer is faster.
The $^{*}$ entries are \emph{time to a BFGS precision-loss warning},
not a successful solve at its requested gradient tolerance.
All seven $T=0$, $B=3$ analytic runs report that warning:
their final gradient is about $1.6\times10^{-8}$, just above
\texttt{gtol=1e-8}. The physical residual is already below
$1.7\times10^{-7}$. The additional line-search work is therefore material
to the measured cost and is not silently discarded.

For the representative strict runs, impurity-solve counts are
\[
\begin{array}{crrr}
 (T,B)&\text{DIIS}&\text{FD}&\text{analytic}\\
 (0,1)&8&18&6\\
 (0,3)&9&154&101^{*}\\
 (0.2,1)&9&15&5\\
 (0.2,3)&6&161&23 .
\end{array}
\]
At zero temperature the analytic version also performs one projected
response solve per impurity ground-state solve.
All returned points, including the warning case, pass the common
post-checks: maximum complete matrix residual below $10^{-6}$ and absolute
errors in the checked physical quantities below $2\times10^{-6}$.
Observed maxima are approximately $1.7\times10^{-7}$ and
$2.0\times10^{-7}$, respectively.
DIIS generally returns much tighter residuals.
Its internal error norm and BFGS's gradient norm are different quantities;
these timings do not claim identical final residuals.

A separate diagnostic reruns only $T=0$, $B=3$ at
\texttt{gtol=1e-7}, leaving the DIIS and inner-fit tolerances unchanged.
All seven runs of each method then finish successfully:
\begin{center}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Method & Median time (ms) & Impurity solves & Relative to DIIS\\
\midrule
DIIS & 29.48 & 9 & 1.00\\
BFGS, FD response & 349.31 & 133 & 11.85\\
BFGS, analytic adjoint & 174.43 & 19 & 5.92\\
\bottomrule
\end{tabular}
\end{center}
The analytic diagnostic additionally uses 19 projected response solves.
Its physical residual is below $3.1\times10^{-7}$ and the checked observable
errors are below $6.7\times10^{-8}$.
This is a different requested gradient tolerance; it is not presented as
a successful \texttt{gtol=1e-8} result.

The practical observation is that analytic minimization wins in the tiny
$B=1$ examples, whereas current DIIS is faster for the tested $B=3$ cases.
The strict BFGS warning also demonstrates sensitivity to inner-solver
initialization and energy-line-search precision near a flat minimum.
Neither the earlier adjoint speedup over finite differences nor these
DIIS ratios establish a universal optimizer ranking.

\subsection{What is and is not general}
The chain-rule and Schur-complement derivations extend to multiple clusters,
rectangular ghost matrices, and complex entries represented by real
coordinates, provided the actual upfolding and constraint maps are used.
The single-impurity checks cover one orbital, a paramagnet, half filling,
$U=1$, $B=1,3$, and $T=0,0.2$. The additional coupled two-impurity
check uses $U_1=1$, $U_2=1.4$, $B=1$, the same symmetries, and the same
two temperatures. No minimization test for a CSC DFT interface,
general complex multi-orbital input, or general fixed filling away from
particle--hole symmetry has been performed.

The minimum formulation is not automatically equivalent to every problem
or stationary branch accessible to the root solver. Required conditions
and failure modes include:
\begin{itemize}
 \item Fixed, consistent Hamiltonian and sector; a density-dependent
       double-counting potential needs its scalar energy functional.
 \item Lattice and embedding representability of trial targets.
 \item Ground-state degeneracies, level crossings, pinned density
       eigenvalues, Mott limits, and frozen ghost orbitals.
 \item Gauge freedom, null impurity response, and nonunique inner branches.
 \item Nonsingular quadratic response for thermal elimination; soft modes
       can spoil the implicit-function construction.
 \item Physical instabilities and competing phases: root finding may
       intentionally target a saddle that energy minimization excludes.
 \item Accurate impurity free energies and consistent thermal truncation;
       density matrices alone do not suffice for an energy line search.
\end{itemize}

An energy line search can reject uphill steps on a valid reduced branch,
but it cannot repair a failed inner solve, a singular coordinate map, or a
wrong branch. No broad robustness comparison has been performed.
In particular, this moderate-temperature check does not resolve the
project's separate high-temperature near-flat ghost-solution questions.


\subsection{Comparison with VASP electronic minimization}
\label{sec:vasp-comparison}
VASP distinguishes iterative diagonalization plus density mixing from
direct optimization of the electronic orbitals; see its
\href{https://vasp.at/wiki/Category:Electronic_minimization}{electronic-minimization overview}.
In the SCF route, blocked Davidson (\texttt{ALGO=Normal}) or RMM-DIIS
solves the orbital problem for a supplied Hamiltonian. A separate
Broyden/Pulay density mixer updates the input density. Thus orbital
RMM-DIIS and DIIS-like density mixing are separate algorithmic layers;
SCF is not synonymous with one particular DIIS solver.

For direct optimization,
\href{https://vasp.at/wiki/IALGO}{VASP's documented algorithms}
include preconditioned all-band conjugate gradients
(\texttt{ALGO=All} or \texttt{Conjugate}, \texttt{IALGO=58}) and damped
orbital dynamics (\texttt{ALGO=Damped}, \texttt{IALGO=53}).
These update orbitals and occupations, respecting orbital orthonormality,
and use
the self-consistent total free energy as their variational quantity.
They differ from our generic BFGS optimizer over embedding fields.
VASP recommends an improved line search,
\href{https://vasp.at/wiki/ISEARCH}{\texttt{ISEARCH=1}}, for \texttt{ALGO=All}.
The thermal/smearing occupations belong to VASP's functional; this is not
an interacting impurity Gibbs trace as in thermal RISB theory.

Direct optimization changes how self-consistency is reached: it supplies
an energy-based search direction and updates the density from the current
orbitals, normally without an independent density mixer. This can improve
convergence where a density fixed-point cycle struggles, including certain
orbital-dependent functionals. It does not change the desired physical
stationary equations or guarantee a global minimum. VASP documents
sensitivity to metallic occupations, step size, and energy noise. Its
\href{https://vasp.at/wiki/Direct_optimization_of_the_orbitals}{direct-optimization derivation}
also explains the coupled orbital-rotation and occupation updates.

Our reduction has the analogous aim of using energy or free energy to
guide admissible updates rather than accepting a complete fixed-point
cycle. However, it optimizes embedding Hamiltonian fields and retains inner
solves. Its singular density normalization, decoupled ghosts, degeneracies,
and potentially indefinite thermal reduced curvature remain additional
issues. VASP's experience is motivation for testing the approach, not
evidence that it fixes those RISB failures.

VASP uses several distinct controls that are sometimes called damping.
Its \href{https://vasp.at/wiki/IMIX}{density mixer} includes amplitude control
and Kerker preconditioning, which suppresses problematic long-wavelength
charge updates. Its conjugate-gradient direct optimizer uses a line search.
Its \texttt{Damped} method uses friction in fictitious orbital dynamics;
\href{https://vasp.at/wiki/TIME}{\texttt{TIME}} controls the integration step.
A schematic constrained equation is
$M\ddot\psi+\gamma\dot\psi=-\nabla_{\psi^*}F+$ constraint forces.
These mechanisms share the aim of stabilizing updates. They are not an
identification of VASP's algorithm with the RISB Jacobian-SVD filter in
Eq.~\eqref{eq:damped-matching}.

\section{Applicability to charge-self-consistent DFT+RISB}
\subsection{Replacing the correlated inner solve}
There are two distinct loops. The current correlated solve holds
\path{h0_k}, \path{projectors}, \path{gf_struct_mapping}, and the interactions
fixed. The CSC loop changes the DFT charge, and hence the next Kohn--Sham
Hamiltonian and possibly the projectors and basis.

The multi-impurity minimizer can serve as an alternative correlated
inner solver at each fixed DFT update, subject to the representability and
regularity conditions already stated. This does not turn the outer CSC
iteration into direct minimization of one joint DFT+RISB functional.
A decrease of the inner objective is only a decrease for that fixed
Hamiltonian and fixed subspaces; it is not a monotonicity statement between
successive DFT updates.

The output density must be the physical density, using all clusters:
\begin{lstlisting}[language=Python]
density_k = helpers.get_density_matrix_k(
    R_full=S.R_full,
    vectors=S.bloch_vector_qp,
    kweights=S.kweights,
    rho_c=S.rho_c,
    projectors=S.projectors,
    gf_struct_mapping=S.gf_struct_mapping,
)
\end{lstlisting}
The inner solver must populate these attributes from the same accepted
trial. This helper includes the impurity local density correction;
using only the coherent $R^\dagger\rho_{\qp}R$ contribution is insufficient.
Use \path{density_k} or its difference from \path{density_dft_k} in the
DFT charge interface. Only the total electron count is prescribed;
charges on inequivalent impurities remain variational.

\subsection{Fixed-DFT energy and double counting}
At zero temperature the repository's energy bookkeeping is
\begin{align}
 E_{\rm DFT+RISB}
 &= E_{\rm DFT}-E_{\rm band}^{\rm DFT}
       +E_{\rm band}^{\rm RISB}\notag\\
 &\quad+\sum_i\langle H_{{\rm int},i}\rangle
       -\sum_i E_{{\rm dc},i}[\rho_{c,i}].
 \label{eq:csc-energy}
\end{align}
Evaluate this total with \texttt{get\_total\_energy}.
Use \texttt{get\_band\_energy} for both band contributions.
In this CSC decomposition \path{h_int[i]} contains the local interaction;
the local Kohn--Sham quadratic terms are supplied separately by the solver.
At fixed DFT input, the first two terms are constant.
The double-counting energy must be consistent with the
\path{double_counting(i, rho_c)} potential and the stored transpose convention.
An arbitrary potential callback need not possess the required scalar energy.

In \path{BaseLatticeSolver.solve_impurities}, the local variable
\path{potential} is subtracted from \path{h0_loc_matrix[i]} before
\texttt{set\_h\_imp}. Consequently,
\path{imp.h_int + imp.h0_loc} is the \emph{shifted} local Hamiltonian.
Its expectation alone is not the physical local contribution in
Eq.~\eqref{eq:csc-energy}. For a fixed trial potential, the correction is
\begin{equation}
 \begin{split}
 E_{\rm fixed\ KS}
 &=K+\sum_i \texttt{h\_loc\_energy}[i]\\
 &\quad+\sum_i\mathrm{potential}_i:\rho_{c,i}
             -\sum_i E_{{\rm dc},i}[\rho_{c,i}].
 \end{split}
 \label{eq:csc-addback}
\end{equation}
The add-back removes the potential already subtracted from the impurity
Hamiltonian; the scalar double-counting energy is then subtracted once.
This avoids substituting a potential expectation for the double-counting
energy itself.

A constructive extension of the zero-temperature impurity-solve
parameterization is to treat these local \path{potential} matrices as
additional independent trial fields:
\begin{equation}
 H_{{\emb},i}
 =H_{{\loc},i}^{\rm KS}+H_{{\rm int},i}
  -\mathrm{potential}_i:\hat\rho_{c,i}
  +\sum_a y_{ia}A_{ia}.
\end{equation}
Here $H_{{\loc},i}^{\rm KS}$ uses the original, unshifted local Kohn--Sham
matrix. The trial \path{potential} is a new optimizer coordinate, not an
additional physical interaction. Define the consistent double-counting
potential by
\[
 \dd E_{{\rm dc},i}
 =\mathrm{potential}_{{\rm dc},i}:\dd\rho_{c,i}.
\]
Hellmann--Feynman applied to Eq.~\eqref{eq:csc-addback} then gives
\begin{equation}
 \begin{split}
 \dd E_{\rm fixed\ KS}
 &=\langle\dd_R\hat T_{\{R_i\}}\rangle
   -\sum_i\bigl(\Lambda_i:\dd\rho_{f,i}
                    +y_i\cdot\dd m_i\bigr)\\
 &\quad+\sum_i
   (\mathrm{potential}_i-\mathrm{potential}_{{\rm dc},i})
            :\dd\rho_{c,i}.
 \end{split}
 \label{eq:csc-gradient}
\end{equation}
Thus an integrable density-dependent double counting adds a definite
physical-density response term. On a sufficiently responsive, regular
branch, stationarity recovers the corresponding local potential condition,
together with the earlier embedding equations.

This extension is derived here but is not implemented or tested against a
DFT code. The simple fixed-\path{h0_loc_matrix} pseudocode must be extended:
the present callback inserts a potential from the previous impurity density,
whereas this variational parameterization supplies an independent trial
potential and evaluates its scalar energy and derivative consistently.
A step that changes that density-dependent potential cannot ignore the last
line of Eq.~\eqref{eq:csc-gradient}.


\subsection{Meaning of the potential callback}
The callback is simply the supplied Python function
\path{double_counting(i, rho_c)}. It returns a potential matrix for the local
Hamiltonian. A minimizer also needs the corresponding scalar correction
energy to compare trial states consistently:
\[
 \dd E_{\rm dc}
 =\mathrm{potential}_{\rm dc}:\dd\rho_c.
\]
For a scalar occupation,
\[
 E_{\rm dc}(n)=\frac{\alpha}{2}n^2,\qquad
 \mathrm{potential}_{\rm dc}(n)=\alpha n.
\]
The product $\mathrm{potential}_{\rm dc}n=\alpha n^2$ is twice the energy
in this example. One cannot generally substitute it for $E_{\rm dc}$.
A known prescription may already determine both quantities; the interface
must expose or calculate them consistently. An arbitrary potential callback
need not be the derivative of any scalar functional.

\subsection{Finite-temperature double-counting extension}
The earlier note wrote out the zero-temperature correction explicitly.
There is no fundamental restriction to zero temperature.
At fixed DFT input, let
$\mathcal L_T^{\rm base}(x,y,\{\mathrm{potential}_i\})$
be the implemented thermal functional with
\path{h0_loc_matrix[i] - potential[i]} in the interacting impurity.
The trial potentials do not modify the quadratic-reference term independently.

Introduce physical density targets $\rho_{c,i}$ and define
\begin{equation}
 \boxed{\begin{aligned}
 \mathcal L_T^{\rm dc}
 &=\mathcal L_T^{\rm base}(x,y,\{\mathrm{potential}_i\})\\
 &\quad+\sum_i\bigl[
   \mathrm{potential}_i:\rho_{c,i}
                   -E_{{\rm dc},i}(\rho_{c,i})\bigr].
 \end{aligned}}
 \label{eq:thermal-dc-functional}
\end{equation}
At this stage the target \path{rho_c} is an independent density coordinate.
Variation of the actual impurity grand potential gives
\begin{align}
 \partial_{\mathrm{potential}_i}\mathcal L_T^{\rm dc}
 &=\rho_{c,i}-\langle\hat\rho_{c,i}\rangle_{\emb,i}=0,
 \label{eq:thermal-dc-density}\\
 \partial_{\rho_{c,i}}\mathcal L_T^{\rm dc}
 &=\mathrm{potential}_i
      -\partial_{\rho_{c,i}}E_{{\rm dc},i}=0.
 \label{eq:thermal-dc-potential}
\end{align}
Thus the density equals \texttt{get\_rho\_c} and the Hamiltonian potential
equals the chosen double-counting derivative. The derivatives in $x,y$
remain the corresponding thermal embedding stationarity equations with
these local fields. This is a direct finite-temperature stationarity
construction, not a guessed entropy correction.

A useful elimination is the constrained impurity free energy
\begin{equation}
 \Gamma_{\emb,i}(y_i,\rho_{c,i})
 =\sup_{\mathrm{potential}_i}
 \left[
 \Omega_{\emb,i}(y_i,\mathrm{potential}_i)
       +\mathrm{potential}_i:\rho_{c,i}\right].
\end{equation}
Its inner equation is Eq.~\eqref{eq:thermal-dc-density}. Concavity of the
impurity grand potential in a linear field makes this a maximization,
on the nonredundant representable coordinates.
The remaining functional contains
$\Gamma_{\emb,i}-E_{{\rm dc},i}$, whose derivative with respect to
$\rho_{c,i}$ is the potential mismatch in
Eq.~\eqref{eq:thermal-dc-potential}.
The same reduced-curvature analysis applies afterward; a scalar
double-counting functional does not by itself establish global convexity.

The finite-temperature algebra was checked by differentiating an actual
$B=3$ impurity grand potential at $T=0.5$, with a nonzero local potential
and a quadratic scalar double counting. The two derivatives of the
augmented functional agree with
Eqs.~\eqref{eq:thermal-dc-density}--\eqref{eq:thermal-dc-potential}
within $10^{-8}$. This is a derivative check, not a full double-counting
CSC minimization.

If the chosen correction itself depends on temperature, replace
$E_{\rm dc}$ by the specified free-energy contribution and include its
temperature derivative in entropy bookkeeping.
The full DFT charge/orbital optimization still needs its own coupled
functional and derivatives.

\subsection{What a complete CSC minimum would additionally require}
A practical nested workflow is:
\begin{lstlisting}[language=Python]
dft = run_dft()  # Supplied by the external DFT interface.
for charge_iteration in range(max_charge_iterations):
    S = prepare_risb(dft)  # Every correlated cluster.

    # NEW inner optimizer, including consistent double
    # counting and the prescribed total physical filling.
    minimize_fixed_h0(S, double_counting_energy=...)

    density_k = helpers.get_density_matrix_k(...)
    total_energy = helpers.get_total_energy(...)
    dft_next = update_dft_charge(dft, density_k)

    if charge_energy_and_risb_converged(...):
        break
    dft = dft_next
\end{lstlisting}
The external DFT functions are schematic, as in
\path{docs/how-to/charge_self_consistency.md}. The RISB density and energy
helper names are actual repository functions.

To derive and implement direct minimization of the \emph{entire} CSC problem,
one must also specify the charge/orbital variables, the DFT functional and its
response, the charge-consistency constraints, and any dependence of the
projectors, interaction parameters, and basis on those variables.
Their derivative terms must be included. Changing projectors during an
inner fixed-subspace line search is not covered by the earlier gradient.

At finite temperature one additionally needs a consistent total
DFT+RISB free-energy functional, common temperature and chemical-potential
conventions, and entropy bookkeeping that does not count the DFT band
entropy twice. \texttt{get\_total\_energy} is an energy bookkeeping helper;
it does not implement that general thermal functional.
The finite-temperature double-counting stationarity extension is derived
above and its local derivatives have been checked. No complete joint CSC
minimization against a DFT code has been run.


\paragraph{Meaning of derived and checked.}
The identities are algebraically derived under their stated conditions.
Numerical cross-checks cover the specified impurity derivatives and local
minimizations, including the weighted adjoint and current DIIS comparisons.
A complete CSC calculation, a degenerate Mott-state adjoint minimization,
general complex multi-orbital minimization, and a broad scaling/robustness
study have not been run. Agreement with the existing root solver checks
algorithms within RISB theory; it is not a comparison with an exact lattice solution.
A derivative check must not be described as a full minimization check.

\paragraph{Compilation.}
Create \path{_build/energy_minimization}; from the repository root, run twice:
{\scriptsize
\begin{verbatim}
pdflatex -output-directory=_build/energy_minimization notes/energy_minimization.md
\end{verbatim}
}
\end{document}
