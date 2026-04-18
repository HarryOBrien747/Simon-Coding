! ============================================================================
! cluster_bhtree_yoshida6.f90
! Barnes-Hut octree (monopole) + Yoshida 6th-order symplectic integrator, OpenMP.
!
! Integrator: Yoshida (1990) 6th-order symmetric composition
!   - 7 force evaluations per step, 8 drifts
!   - Error scales as O(dt^6) vs O(dt^2) leapfrog, O(dt^4) Yoshida-4
!   - Ideal when you want very tight energy conservation with moderate dt
!
! With dt=0.0002 Myr over 2 Myr:
!   - nsteps = 10,000
!   - force evals = 10,000 x 7 = 70,000
!   - NOTE: for N=100,000 this is a serious HPC run. Each force eval is
!     O(N log N) ~ 1.7M ops. Budget several hours depending on thread count.
!     Use as many OpenMP threads as your machine has (export OMP_NUM_THREADS=N).
!
! Compile:
!   gfortran -O3 -march=native -fopenmp -o cluster_yoshida6 cluster_bhtree_yoshida6.f90
!
! Run:
!   export OMP_NUM_THREADS=16   # (or however many cores you have)
!   ./cluster_yoshida6 100000
!
! Outputs:
!   energy_YYYYMMDD_HHMMSS.txt          : t[s], E_tree[J], dE_tree/E0_tree
!   energy_paircheck_YYYYMMDD_HHMMSS.txt: (only if N <= 5000)
!   snaps_YYYYMMDD_HHMMSS.bin           : binary snapshots (N32, t, x,y,z in pc as float32)
!   masses_YYYYMMDD_HHMMSS.txt          : one mass per line [Msun]
!   masshist_YYYYMMDD_HHMMSS.txt        : mass histogram
! ============================================================================
program cluster_bhtree_yoshida6
  use, intrinsic :: iso_fortran_env, only: int32, real32, real64, output_unit
  use omp_lib
  implicit none

  integer, parameter :: dp = real64
  real(dp), parameter :: G    = 6.67430e-11_dp
  real(dp), parameter :: Msun = 1.98847e30_dp
  real(dp), parameter :: pc   = 3.0856775814913673e16_dp
  real(dp), parameter :: Myr  = 1.0e6_dp * 365.25_dp * 86400.0_dp

  ! ---- Cluster parameters ----
  real(dp), parameter :: Rscale_pc = 1.0_dp   ! Plummer scale length [pc]
  real(dp), parameter :: Q_target  = 1.0_dp   ! virial ratio 2T/|U|

  ! ---- Force accuracy ----
  ! theta=0.1 is a good balance for N=100k (tighter costs more but small benefit)
  real(dp), parameter :: theta_force  = 0.1_dp
  real(dp), parameter :: theta_energy = 0.02_dp  ! very tight; only used at output times

  ! ---- Softening ----
  real(dp), parameter :: eps_m = 1.0e15_dp   ! [m]; ~0.032 pc, reasonable for 100k

  ! ---- Run settings (as requested) ----
  real(dp), parameter :: t_end_Myr = 0.01_dp     ! total run time [Myr]
  real(dp), parameter :: dt_Myr    = 0.0002_dp  ! timestep [Myr]
  real(dp), parameter :: out_Myr   = 0.02_dp    ! snapshot/energy cadence [Myr]
  !
  ! With these settings:
  !   nsteps      = 2.0 / 0.0002 = 10,000
  !   write_every = 0.02 / 0.0002 = 100 steps
  !   n_outputs   = 100
  !   force_evals = 10,000 x 7 = 70,000  (6th-order Yoshida uses 7/step)

  integer, parameter :: seed0 = -12345

  ! Pairwise check only for small N
  integer, parameter :: N_paircheck_max = 5000

  ! ================================================================
  ! Yoshida 6th-order symplectic coefficients
  ! Reference: Yoshida, H. (1990), Phys. Lett. A, 150, 262-268, Solution A
  !
  ! Build the 6th-order integrator by tripling the 4th-order composition:
  !   z1 = 1/(2 - 2^(1/5))       (outermost scaling)
  !   z0 = 1 - 2*z1
  !
  ! The 7 kick (d) and 8 drift (c) coefficients follow from the palindrome:
  !   d = [w3, w2, w1, w0, w1, w2, w3]
  !   c = [w3/2, (w3+w2)/2, (w2+w1)/2, (w1+w0)/2,   <- first half
  !              (w0+w1)/2, (w1+w2)/2, (w2+w3)/2, w3/2]  <- mirror
  ! ================================================================
  real(dp), parameter :: yo6_w1 =  0.784513610477557803_dp
  real(dp), parameter :: yo6_w2 =  0.235573213359358134_dp
  real(dp), parameter :: yo6_w3 = -1.17767998417887100_dp
  real(dp), parameter :: yo6_w0 =  1.0_dp - 2.0_dp*(yo6_w1 + yo6_w2 + yo6_w3)
  !   yo6_w0 = 1.315186320683906 (positive, as it should be)

  ! 7 kick coefficients
  real(dp), parameter :: yd(7) = [ yo6_w3, yo6_w2, yo6_w1, yo6_w0, &
                                    yo6_w1, yo6_w2, yo6_w3 ]

  ! 8 drift coefficients (palindrome)
  real(dp), parameter :: yc(8) = [ yo6_w3/2.0_dp, &
                                    (yo6_w3+yo6_w2)/2.0_dp, &
                                    (yo6_w2+yo6_w1)/2.0_dp, &
                                    (yo6_w1+yo6_w0)/2.0_dp, &
                                    (yo6_w0+yo6_w1)/2.0_dp, &
                                    (yo6_w1+yo6_w2)/2.0_dp, &
                                    (yo6_w2+yo6_w3)/2.0_dp, &
                                    yo6_w3/2.0_dp ]

  ! ---- State ----
  integer :: N, step, nsteps, write_every, every_print, sub
  real(dp) :: t, dt, t_end, eps2, E0, Ecur, dErel
  real(dp) :: E0_pair, Ecur_pair, rel_tree_pair
  logical  :: do_paircheck

  real(dp), allocatable :: m(:), r(:,:), v(:,:), accel(:,:)
  integer :: kdum

  ! Tree storage
  integer :: max_nodes, node_count
  integer(int32), allocatable :: body(:), child(:,:)
  real(dp), allocatable :: mass(:), half(:), com(:,:), cen(:,:)

  ! Output
  character(len=32)  :: tag
  character(len=128) :: energy_name, energy_check_name, snaps_name
  integer, parameter :: uener=11, usnap=12, ucheck=13
  integer :: ios

  ! ---- Setup ----
  call get_N_from_cli(N)
  if (N <= 0) N = 10000  ! default 100k

  dt    = dt_Myr * Myr
  t_end = t_end_Myr * Myr
  eps2  = eps_m * eps_m

  allocate(m(N), r(3,N), v(3,N), accel(3,N))

  max_nodes = 8*N + 16
  allocate(body(max_nodes), child(8,max_nodes), mass(max_nodes), &
           half(max_nodes), com(3,max_nodes), cen(3,max_nodes))

  do_paircheck = (N <= N_paircheck_max)

  ! ---- Initial conditions ----
  kdum = seed0
  call make_cluster_ic_plummer(N, Rscale_pc*pc, Q_target, kdum, m, r, v)
  call center_on_com(N, m, r, v)

  ! Build tree + initial energy (accel not needed before first step in Yoshida)
  call build_tree(N, r, max_nodes, node_count, body, child, cen, half)
  call compute_multipoles(N, m, r, node_count, body, child, mass, com)

  E0 = total_energy_tree(N, m, r, v, node_count, body, child, mass, com, half, eps2, theta_energy)
  if (do_paircheck) then
    E0_pair = total_energy_pairwise(N, m, r, v, eps2)
  else
    E0_pair = 0.0_dp
  end if

  ! ---- Open output files ----
  call timestamp_tag(tag)
  energy_name       = 'energy_'       //trim(tag)//'.txt'
  energy_check_name = 'energy_paircheck_'//trim(tag)//'.txt'
  snaps_name        = 'snaps_'        //trim(tag)//'.bin'

  call write_masses(tag, N, m, Msun)
  call write_mass_bins(tag, N, m, Msun)

  open(uener, file=energy_name, status='replace', action='write', form='formatted', iostat=ios)
  if (ios /= 0) stop 'Cannot open energy file.'
  write(uener,'(A)') 't[s]  E_tree[J]  dE_tree/E0_tree'

  if (do_paircheck) then
    open(ucheck, file=energy_check_name, status='replace', action='write', form='formatted', iostat=ios)
    if (ios /= 0) stop 'Cannot open paircheck file.'
    write(ucheck,'(A)') 't[s]  E_tree[J]  E_pair[J]  (E_tree-E_pair)/E_pair'
  end if

  open(usnap, file=snaps_name, status='replace', access='stream', form='unformatted', action='write', iostat=ios)
  if (ios /= 0) stop 'Cannot open snapshot file.'

  ! Write t=0
  t = 0.0_dp
  call write_energy(uener, t, E0, 0.0_dp)
  call write_snapshot_bin(usnap, N, t, r, pc)
  if (do_paircheck) then
    rel_tree_pair = (E0 - E0_pair) / E0_pair
    write(ucheck,'(ES25.16,3(1X,ES25.16))') t, E0, E0_pair, rel_tree_pair
  end if

  nsteps      = int(t_end / dt)
  write_every = max(1, int((out_Myr*Myr) / dt))
  every_print = max(1, int(0.01_dp * nsteps))

  write(*,'(A)')        '============================================='
  write(*,'(A,I0)')     'N                = ', N
  write(*,'(A,I0)')     'OpenMP threads   = ', omp_get_max_threads()
  write(*,'(A)')        '--- Yoshida 6th-Order Symplectic Integrator ---'
  write(*,'(A)')        '    7 force evals per step, 8 drifts'
  write(*,'(A,ES12.4)') 'dt       [Myr]   = ', dt/Myr
  write(*,'(A,ES12.4)') 't_end    [Myr]   = ', t_end/Myr
  write(*,'(A,I0)')     'nsteps           = ', nsteps
  write(*,'(A,I0)')     'total force evals= ', nsteps*7
  write(*,'(A,ES12.4)') 'theta_force      = ', theta_force
  write(*,'(A,ES12.4)') 'theta_energy     = ', theta_energy
  write(*,'(A,ES12.4)') 'eps      [m]     = ', eps_m
  write(*,'(A,ES12.4)') 'yo6_w0           = ', yo6_w0
  write(*,'(A,ES12.4)') 'yo6_w1           = ', yo6_w1
  write(*,'(A,ES12.4)') 'yo6_w2           = ', yo6_w2
  write(*,'(A,ES12.4)') 'yo6_w3           = ', yo6_w3
  write(*,'(A,ES12.4)') 'sum(yc)          = ', sum(yc)   ! should be 1.0
  write(*,'(A,ES12.4)') 'sum(yd)          = ', sum(yd)   ! should be 1.0
  write(*,'(A,ES12.4)') 'E0_tree   [J]    = ', E0
  if (do_paircheck) write(*,'(A,ES12.4)') 'E0_pair   [J]    = ', E0_pair
  if (do_paircheck) then
    write(*,'(A)') 'Pairwise check: ON'
  else
    write(*,'(A)') 'Pairwise check: OFF (N > 5000)'
  end if
  write(*,'(A)') '============================================='

  call show_progress(0, nsteps)

  ! ================================================================
  ! MAIN LOOP - Yoshida 6th-order
  !
  ! Structure per step (palindrome, 8 drifts, 7 kicks):
  !   drift yc(1)*dt
  !   force -> kick yd(1)*dt          <- yc(1), yd(1) use wo6_w3 (negative! OK)
  !   drift yc(2)*dt
  !   force -> kick yd(2)*dt
  !   drift yc(3)*dt
  !   force -> kick yd(3)*dt
  !   drift yc(4)*dt
  !   force -> kick yd(4)*dt          <- central kick, yd(4) uses yo6_w0 (large positive)
  !   drift yc(5)*dt   [= yc(4)]
  !   force -> kick yd(5)*dt   [= yd(3)]
  !   drift yc(6)*dt   [= yc(3)]
  !   force -> kick yd(6)*dt   [= yd(2)]
  !   drift yc(7)*dt   [= yc(2)]
  !   force -> kick yd(7)*dt   [= yd(1)]
  !   drift yc(8)*dt   [= yc(1)]
  !
  ! NOTE: yc(1), yc(2), yd(1) are negative (yo6_w3 < 0).
  !       This is mathematically correct - particles drift briefly backward.
  !       Do NOT be alarmed by this; it is intrinsic to all high-order
  !       Yoshida methods and is what gives 6th-order accuracy.
  ! ================================================================
  do step = 1, nsteps

    do sub = 1, 7
      ! Drift
      r = r + yc(sub) * dt * v

      ! Rebuild tree at new positions and compute forces
      call build_tree(N, r, max_nodes, node_count, body, child, cen, half)
      call compute_multipoles(N, m, r, node_count, body, child, mass, com)
      call tree_accel_all(N, r, node_count, body, child, mass, com, half, eps2, theta_force, accel)

      ! Kick
      v = v + yd(sub) * dt * accel
    end do

    ! Final drift (sub=8, closes the palindrome)
    r = r + yc(8) * dt * v

    t = t + dt

    ! ---- Output ----
    if (mod(step, write_every) == 0) then
      ! Rebuild tree at current r for energy diagnostic
      call build_tree(N, r, max_nodes, node_count, body, child, cen, half)
      call compute_multipoles(N, m, r, node_count, body, child, mass, com)
      Ecur  = total_energy_tree(N, m, r, v, node_count, body, child, mass, com, half, eps2, theta_energy)
      dErel = (Ecur - E0) / E0
      call write_energy(uener, t, Ecur, dErel)
      call write_snapshot_bin(usnap, N, t, r, pc)

      if (do_paircheck) then
        Ecur_pair     = total_energy_pairwise(N, m, r, v, eps2)
        rel_tree_pair = (Ecur - Ecur_pair) / Ecur_pair
        write(ucheck,'(ES25.16,3(1X,ES25.16))') t, Ecur, Ecur_pair, rel_tree_pair
      end if
    end if

    if (mod(step, every_print) == 0) call show_progress(step, nsteps)

  end do

  call show_progress(nsteps, nsteps)
  write(*,*)
  close(uener); close(usnap)
  if (do_paircheck) close(ucheck)

  write(*,'(A)') 'Done.'
  write(*,'(A)') 'Energy : '//trim(energy_name)
  if (do_paircheck) write(*,'(A)') 'Pair   : '//trim(energy_check_name)
  write(*,'(A)') 'Snaps  : '//trim(snaps_name)

contains

  ! ============================================================
  ! Mass histogram
  ! ============================================================
  subroutine write_mass_bins(tag, N, m_kg, Msun_)
    character(len=*), intent(in) :: tag
    integer,  intent(in) :: N
    real(dp), intent(in) :: m_kg(N), Msun_
    real(dp), parameter :: mmin=0.2_dp, mmax=4.0_dp, dm=0.2_dp
    integer,  parameter :: nbin = nint((mmax-mmin)/dm)
    integer  :: counts(nbin), i, b, u
    real(dp) :: ms, lo, hi, pct
    character(len=128) :: fname

    counts = 0
    do i = 1, N
      ms = m_kg(i)/Msun_
      b  = int((ms-mmin)/dm) + 1
      b  = max(1, min(nbin, b))
      counts(b) = counts(b) + 1
    end do

    fname = 'masshist_'//trim(tag)//'.txt'
    u = 20
    open(u, file=fname, status='replace', action='write', form='formatted')
    write(u,'(A)') 'm_lo  m_hi  count  percent'
    do b = 1, nbin
      lo  = mmin + (b-1)*dm
      hi  = lo + dm
      pct = 100.0_dp * real(counts(b),dp) / real(N,dp)
      write(u,'(F5.1,1X,F5.1,1X,I9,1X,F10.6)') lo, hi, counts(b), pct
    end do
    close(u)
    write(*,'(A)') 'Mass histogram: masshist_'//trim(tag)//'.txt'
  end subroutine write_mass_bins

  ! ============================================================
  ! CLI argument
  ! ============================================================
  subroutine get_N_from_cli(N_out)
    integer, intent(out) :: N_out
    character(len=64) :: arg
    integer :: stat
    N_out = -1
    call get_command_argument(1, arg, status=stat)
    if (stat == 0) then
      read(arg,*,iostat=stat) N_out
      if (stat /= 0) N_out = -1
    end if
  end subroutine get_N_from_cli

  ! ============================================================
  ! Progress bar
  ! ============================================================
  subroutine show_progress(i, total)
    integer, intent(in) :: i, total
    integer, parameter  :: width = 50
    real(dp) :: frac
    integer  :: filled
    character(len=width) :: bar
    frac   = real(i,dp) / max(1, total)
    filled = max(0, min(width, int(frac*width)))
    bar    = repeat('#', filled) // repeat('-', width-filled)
    write(output_unit,'(A)',advance='no') &
      achar(13)//'['//bar//'] '//trim(adjustl(pct_str(100.0_dp*frac)))//'%'
    call flush(output_unit)
  end subroutine show_progress

  pure function pct_str(x) result(s)
    real(dp), intent(in) :: x
    character(len=6) :: s
    write(s,'(F5.1)') x
  end function pct_str

  ! ============================================================
  ! Energy / snapshot writers
  ! ============================================================
  subroutine write_energy(u, t_, E, dErel)
    integer,  intent(in) :: u
    real(dp), intent(in) :: t_, E, dErel
    write(u,'(ES25.16,2(1X,ES25.16))') t_, E, dErel
  end subroutine write_energy

  subroutine write_snapshot_bin(u, N_, t_, r_, pc_unit)
    integer,  intent(in) :: u, N_
    real(dp), intent(in) :: t_, r_(3,N_), pc_unit
    integer(int32) :: N32
    real(real32), allocatable, save :: sx(:), sy(:), sz(:)
    integer, save :: nalloc = 0
    integer :: i

    if (nalloc /= N_) then
      if (allocated(sx)) deallocate(sx,sy,sz)
      allocate(sx(N_), sy(N_), sz(N_))
      nalloc = N_
    end if
    do i = 1, N_
      sx(i) = real(r_(1,i)/pc_unit, real32)
      sy(i) = real(r_(2,i)/pc_unit, real32)
      sz(i) = real(r_(3,i)/pc_unit, real32)
    end do
    N32 = int(N_, int32)
    write(u) N32; write(u) t_; write(u) sx; write(u) sy; write(u) sz
  end subroutine write_snapshot_bin

  ! ============================================================
  ! Plummer sphere ICs + Kroupa IMF
  ! ============================================================
  subroutine make_cluster_ic_plummer(N_, Rscale_m, Q, kdum_, m_, r_, v_)
    integer,  intent(in)    :: N_
    real(dp), intent(in)    :: Rscale_m, Q
    integer,  intent(inout) :: kdum_
    real(dp), intent(out)   :: m_(N_), r_(3,N_), v_(3,N_)
    integer :: i
    real(real32), allocatable :: r0(:,:), v0(:,:)
    real(dp) :: T, U, scale

    allocate(r0(3,N_), v0(3,N_))

    do i = 1, N_
      m_(i) = sample_kroupa_02_4(kdum_) * Msun
    end do

    call plummer(N_, r0, v0, kdum_)

    do i = 1, N_
      r_(:,i) = real(r0(:,i),dp) * Rscale_m
      v_(:,i) = real(v0(:,i),dp)
    end do

    call center_on_com(N_, m_, r_, v_)
    call kinetic_potential_pairwise(N_, m_, r_, v_, 0.0_dp, T, U)
    if (T <= 0.0_dp .or. U >= 0.0_dp) stop 'Bad initial energies.'
    scale = sqrt((Q * abs(U)) / (2.0_dp * T))
    v_ = v_ * scale
    call center_on_com(N_, m_, r_, v_)

    deallocate(r0, v0)
  end subroutine make_cluster_ic_plummer

  real(dp) function sample_kroupa_02_4(kdum_)
    integer, intent(inout) :: kdum_
    real(dp), parameter :: m1=0.2_dp, mb=0.5_dp, m2=4.0_dp, a1=1.3_dp, a2=2.3_dp
    real(dp) :: u, w1k, w2k
    w1k = integral_powerlaw(m1, mb, a1)
    w2k = integral_powerlaw(mb, m2, a2)
    u   = ran_u(kdum_)
    if (u < w1k/(w1k+w2k)) then
      sample_kroupa_02_4 = sample_powerlaw(m1, mb, a1, kdum_)
    else
      sample_kroupa_02_4 = sample_powerlaw(mb, m2, a2, kdum_)
    end if
  end function sample_kroupa_02_4

  real(dp) function integral_powerlaw(mmin, mmax, alpha)
    real(dp), intent(in) :: mmin, mmax, alpha
    if (abs(alpha-1.0_dp) < 1.0e-12_dp) then
      integral_powerlaw = log(mmax/mmin)
    else
      integral_powerlaw = (mmax**(1.0_dp-alpha) - mmin**(1.0_dp-alpha)) / (1.0_dp-alpha)
    end if
  end function integral_powerlaw

  real(dp) function sample_powerlaw(mmin, mmax, alpha, kdum_)
    real(dp), intent(in) :: mmin, mmax, alpha
    integer,  intent(inout) :: kdum_
    real(dp) :: u
    u = ran_u(kdum_)
    if (abs(alpha-1.0_dp) < 1.0e-12_dp) then
      sample_powerlaw = mmin * (mmax/mmin)**u
    else
      sample_powerlaw = (u*(mmax**(1.0_dp-alpha) - mmin**(1.0_dp-alpha)) &
                        + mmin**(1.0_dp-alpha))**(1.0_dp/(1.0_dp-alpha))
    end if
  end function sample_powerlaw

  real(dp) function ran_u(kdum_)
    integer, intent(inout) :: kdum_
    real(real32) :: x
    x = ran3(kdum_)
    if (x <= 0.0_real32) x = 1.0e-7_real32
    if (x >= 1.0_real32) x = 1.0_real32 - 1.0e-7_real32
    ran_u = real(x, dp)
  end function ran_u

  subroutine center_on_com(N_, m_, r_, v_)
    integer,  intent(in)    :: N_
    real(dp), intent(in)    :: m_(N_)
    real(dp), intent(inout) :: r_(3,N_), v_(3,N_)
    integer  :: i
    real(dp) :: rcom(3), vcom(3), Mtot
    Mtot = sum(m_)
    rcom = 0.0_dp; vcom = 0.0_dp
    do i = 1, N_
      rcom = rcom + m_(i)*r_(:,i)
      vcom = vcom + m_(i)*v_(:,i)
    end do
    rcom = rcom/Mtot; vcom = vcom/Mtot
    do i = 1, N_
      r_(:,i) = r_(:,i) - rcom
      v_(:,i) = v_(:,i) - vcom
    end do
  end subroutine center_on_com

  ! ============================================================
  ! Barnes-Hut tree build
  ! ============================================================
  subroutine build_tree(N_, r_, max_n, nc, body_, child_, cen_, half_)
    integer,  intent(in)    :: N_, max_n
    real(dp), intent(in)    :: r_(3,N_)
    integer,  intent(out)   :: nc
    integer(int32), intent(inout) :: body_(max_n), child_(8,max_n)
    real(dp), intent(inout) :: cen_(3,max_n), half_(max_n)
    integer  :: i
    real(dp) :: rmin(3), rmax(3), c(3), h

    rmin = r_(:,1); rmax = r_(:,1)
    do i = 2, N_
      rmin = min(rmin, r_(:,i))
      rmax = max(rmax, r_(:,i))
    end do
    c = 0.5_dp*(rmin+rmax)
    h = 0.5_dp*maxval(rmax-rmin)
    if (h <= 0.0_dp) h = 1.0_dp
    h = 1.001_dp*h

    nc = 1
    body_(1) = 0; child_(:,1) = 0; cen_(:,1) = c; half_(1) = h

    do i = 1, N_
      call insert_body(1, i, r_, max_n, nc, body_, child_, cen_, half_)
    end do
  end subroutine build_tree

  subroutine insert_body(root, ibody, r_, max_n, nc, body_, child_, cen_, half_)
    integer,  intent(in)    :: root, ibody, max_n
    real(dp), intent(in)    :: r_(3,*)
    integer,  intent(inout) :: nc
    integer(int32), intent(inout) :: body_(max_n), child_(8,max_n)
    real(dp), intent(inout) :: cen_(3,max_n), half_(max_n)
    integer :: node, oct, old

    node = root
    do
      if (body_(node) == 0 .and. all(child_(:,node) == 0)) then
        body_(node) = int(ibody, int32); return

      else if (all(child_(:,node) == 0) .and. body_(node) > 0) then
        old = int(body_(node)); body_(node) = -1
        call create_children(node, max_n, nc, body_, child_, cen_, half_)
        call push_into_child(node, old,   r_, max_n, nc, body_, child_, cen_, half_)
        call push_into_child(node, ibody, r_, max_n, nc, body_, child_, cen_, half_)
        return

      else
        oct = octant(cen_(:,node), r_(:,ibody))
        node = int(child_(oct,node))
      end if
    end do
  end subroutine insert_body

  subroutine push_into_child(node, ibody, r_, max_n, nc, body_, child_, cen_, half_)
    integer,  intent(in)    :: node, ibody, max_n
    real(dp), intent(in)    :: r_(3,*)
    integer,  intent(inout) :: nc
    integer(int32), intent(inout) :: body_(max_n), child_(8,max_n)
    real(dp), intent(inout) :: cen_(3,max_n), half_(max_n)
    integer :: oct, cnode
    oct = octant(cen_(:,node), r_(:,ibody))
    cnode = int(child_(oct,node))
    call insert_body(cnode, ibody, r_, max_n, nc, body_, child_, cen_, half_)
  end subroutine push_into_child

  subroutine create_children(node, max_n, nc, body_, child_, cen_, half_)
    integer, intent(in)    :: node, max_n
    integer, intent(inout) :: nc
    integer(int32), intent(inout) :: body_(max_n), child_(8,max_n)
    real(dp), intent(inout) :: cen_(3,max_n), half_(max_n)
    integer  :: o, cnode
    real(dp) :: ch
    ch = 0.5_dp * half_(node)
    do o = 1, 8
      nc = nc + 1
      if (nc > max_n) stop 'Tree overflow: increase max_nodes.'
      cnode = nc
      child_(o,node) = int(cnode, int32)
      body_(cnode) = 0; child_(:,cnode) = 0
      cen_(:,cnode) = cen_(:,node) + ch*[ merge(1,-1,btest(o-1,0)), &
                                            merge(1,-1,btest(o-1,1)), &
                                            merge(1,-1,btest(o-1,2)) ]
      half_(cnode) = ch
    end do
  end subroutine create_children

  integer function octant(c, x)
    real(dp), intent(in) :: c(3), x(3)
    octant = 1 + merge(1,0,x(1)>=c(1)) + 2*merge(1,0,x(2)>=c(2)) + 4*merge(1,0,x(3)>=c(3))
  end function octant

  subroutine compute_multipoles(N_, m_, r_, nc, body_, child_, mass_, com_)
    integer,  intent(in) :: N_, nc
    real(dp), intent(in) :: m_(N_), r_(3,N_)
    integer(int32), intent(in) :: body_(nc), child_(8,nc)
    real(dp), intent(out) :: mass_(nc), com_(3,nc)
    integer  :: node, o, ch, b
    real(dp) :: mt

    mass_ = 0.0_dp; com_ = 0.0_dp

    do node = nc, 1, -1
      if (body_(node) > 0) then
        b = int(body_(node))
        mass_(node) = m_(b); com_(:,node) = r_(:,b)
      else
        mt = 0.0_dp; com_(:,node) = 0.0_dp
        do o = 1, 8
          ch = int(child_(o,node))
          if (ch <= 0) cycle
          mt = mt + mass_(ch)
          com_(:,node) = com_(:,node) + mass_(ch)*com_(:,ch)
        end do
        if (mt > 0.0_dp) then
          com_(:,node) = com_(:,node)/mt; mass_(node) = mt
        else
          mass_(node) = 0.0_dp; com_(:,node) = 0.0_dp
        end if
      end if
    end do
  end subroutine compute_multipoles

  ! ============================================================
  ! Tree force evaluation (OpenMP)
  ! ============================================================
  subroutine tree_accel_all(N_, r_, nc, body_, child_, mass_, com_, half_, eps2_, theta_, a_)
    integer,  intent(in)  :: N_, nc
    real(dp), intent(in)  :: r_(3,N_), mass_(nc), com_(3,nc), half_(nc)
    integer(int32), intent(in) :: body_(nc), child_(8,nc)
    real(dp), intent(in)  :: eps2_, theta_
    real(dp), intent(out) :: a_(3,N_)

    integer, parameter :: max_stack = 8192   ! bigger stack for 100k
    integer  :: i, top, node, o, ch, b, stack(max_stack)
    real(dp) :: dx(3), r2, invr, invr3, s2, th2

    th2 = theta_ * theta_

    !$omp parallel do default(none) &
    !$omp shared(N_,r_,body_,child_,mass_,com_,half_,eps2_,th2,a_) &
    !$omp private(i,top,node,o,ch,b,stack,dx,r2,invr,invr3,s2) &
    !$omp schedule(static)
    do i = 1, N_
      a_(:,i) = 0.0_dp
      top = 1; stack(1) = 1

      do while (top > 0)
        node = stack(top); top = top - 1
        if (mass_(node) == 0.0_dp) cycle

        if (body_(node) > 0) then
          b = int(body_(node))
          if (b == i) cycle
          dx = com_(:,node) - r_(:,i)
          r2 = dx(1)*dx(1)+dx(2)*dx(2)+dx(3)*dx(3)+eps2_
          invr = 1.0_dp/sqrt(r2); invr3 = invr*invr*invr
          a_(:,i) = a_(:,i) + G*mass_(node)*dx*invr3

        else
          dx = com_(:,node) - r_(:,i)
          r2 = dx(1)*dx(1)+dx(2)*dx(2)+dx(3)*dx(3)+eps2_
          s2 = (2.0_dp*half_(node))**2

          if (s2 < th2*r2) then
            invr = 1.0_dp/sqrt(r2); invr3 = invr*invr*invr
            a_(:,i) = a_(:,i) + G*mass_(node)*dx*invr3
          else
            do o = 1, 8
              ch = int(child_(o,node))
              if (ch <= 0) cycle
              if (top >= max_stack) stop 'Stack overflow: increase max_stack.'
              top = top+1; stack(top) = ch
            end do
          end if
        end if
      end do
    end do
    !$omp end parallel do
  end subroutine tree_accel_all

  subroutine tree_potential_all(N_, r_, nc, body_, child_, mass_, com_, half_, eps2_, theta_, phi_)
    integer,  intent(in)  :: N_, nc
    real(dp), intent(in)  :: r_(3,N_), mass_(nc), com_(3,nc), half_(nc)
    integer(int32), intent(in) :: body_(nc), child_(8,nc)
    real(dp), intent(in)  :: eps2_, theta_
    real(dp), intent(out) :: phi_(N_)

    integer, parameter :: max_stack = 8192
    integer  :: i, top, node, o, ch, b, stack(max_stack)
    real(dp) :: dx(3), r2, invr, s2, th2

    th2 = theta_ * theta_

    !$omp parallel do default(none) &
    !$omp shared(N_,r_,body_,child_,mass_,com_,half_,eps2_,th2,phi_) &
    !$omp private(i,top,node,o,ch,b,stack,dx,r2,invr,s2) &
    !$omp schedule(static)
    do i = 1, N_
      phi_(i) = 0.0_dp
      top = 1; stack(1) = 1

      do while (top > 0)
        node = stack(top); top = top - 1
        if (mass_(node) == 0.0_dp) cycle

        if (body_(node) > 0) then
          b = int(body_(node))
          if (b == i) cycle
          dx = com_(:,node) - r_(:,i)
          r2 = dx(1)*dx(1)+dx(2)*dx(2)+dx(3)*dx(3)+eps2_
          phi_(i) = phi_(i) - G*mass_(node)/sqrt(r2)

        else
          dx = com_(:,node) - r_(:,i)
          r2 = dx(1)*dx(1)+dx(2)*dx(2)+dx(3)*dx(3)+eps2_
          s2 = (2.0_dp*half_(node))**2

          if (s2 < th2*r2) then
            phi_(i) = phi_(i) - G*mass_(node)/sqrt(r2)
          else
            do o = 1, 8
              ch = int(child_(o,node))
              if (ch <= 0) cycle
              if (top >= max_stack) stop 'Stack overflow: increase max_stack.'
              top = top+1; stack(top) = ch
            end do
          end if
        end if
      end do
    end do
    !$omp end parallel do
  end subroutine tree_potential_all

  ! ============================================================
  ! Energy routines
  ! ============================================================
  real(dp) function total_energy_tree(N_, m_, r_, v_, nc, body_, child_, &
                                       mass_, com_, half_, eps2_, theta_)
    integer,  intent(in) :: N_, nc
    real(dp), intent(in) :: m_(N_), r_(3,N_), v_(3,N_)
    real(dp), intent(in) :: mass_(nc), com_(3,nc), half_(nc)
    integer(int32), intent(in) :: body_(nc), child_(8,nc)
    real(dp), intent(in) :: eps2_, theta_
    real(dp) :: T, U
    real(dp), allocatable, save :: phi(:)
    integer,  save :: nphi = 0
    integer :: i

    T = 0.0_dp
    do i = 1, N_
      T = T + 0.5_dp*m_(i)*(v_(1,i)*v_(1,i)+v_(2,i)*v_(2,i)+v_(3,i)*v_(3,i))
    end do

    if (nphi /= N_) then
      if (allocated(phi)) deallocate(phi)
      allocate(phi(N_)); nphi = N_
    end if

    call tree_potential_all(N_, r_, nc, body_, child_, mass_, com_, half_, eps2_, theta_, phi)
    U = 0.0_dp
    do i = 1, N_
      U = U + 0.5_dp*m_(i)*phi(i)
    end do

    total_energy_tree = T + U
  end function total_energy_tree

  real(dp) function total_energy_pairwise(N_, m_, r_, v_, eps2_)
    integer,  intent(in) :: N_
    real(dp), intent(in) :: m_(N_), r_(3,N_), v_(3,N_), eps2_
    real(dp) :: T, U
    call kinetic_potential_pairwise(N_, m_, r_, v_, eps2_, T, U)
    total_energy_pairwise = T + U
  end function total_energy_pairwise

  subroutine kinetic_potential_pairwise(N_, m_, r_, v_, eps2_, T, U)
    integer,  intent(in)  :: N_
    real(dp), intent(in)  :: m_(N_), r_(3,N_), v_(3,N_), eps2_
    real(dp), intent(out) :: T, U
    integer  :: i, k
    real(dp) :: dx(3), r2

    T = 0.0_dp
    do i = 1, N_
      T = T + 0.5_dp*m_(i)*(v_(1,i)*v_(1,i)+v_(2,i)*v_(2,i)+v_(3,i)*v_(3,i))
    end do
    U = 0.0_dp
    do i = 1, N_-1
      do k = i+1, N_
        dx = r_(:,k) - r_(:,i)
        r2 = dx(1)*dx(1)+dx(2)*dx(2)+dx(3)*dx(3)+eps2_
        U = U - G*m_(i)*m_(k)/sqrt(r2)
      end do
    end do
  end subroutine kinetic_potential_pairwise

  ! ============================================================
  ! Timestamp
  ! ============================================================
  subroutine timestamp_tag(tag_)
    character(len=*), intent(out) :: tag_
    integer :: v(8)
    call date_and_time(values=v)
    write(tag_,'(I4.4,I2.2,I2.2,"_",I2.2,I2.2,I2.2)') v(1),v(2),v(3),v(5),v(6),v(7)
  end subroutine timestamp_tag

  ! ============================================================
  ! Plummer sampler (unchanged from original)
  ! ============================================================
  subroutine plummer(n_, r_, v_, kdum_)
    integer,     intent(in)    :: n_
    integer,     intent(inout) :: kdum_
    real(real32),intent(out)   :: r_(3,n_), v_(3,n_)
    integer  :: p
    real(real32) :: A(8), ri, twopi
    twopi = 8.0_real32*atan(1.0_real32)
    do p = 1, n_
30    A(1) = ran3(kdum_)
      if (A(1) < 1.0e-10_real32) goto 30
      ri = (A(1)**(-0.6666667_real32) - 1.0_real32)**(-0.5_real32)
      if (ri > 14.6154_real32) goto 30
      A(2) = ran3(kdum_); A(3) = ran3(kdum_)
      r_(3,p) = (1.0_real32 - 2.0_real32*A(2))*ri
      r_(1,p) = sqrt(ri**2 - r_(3,p)**2)*cos(twopi*A(3))
      r_(2,p) = sqrt(ri**2 - r_(3,p)**2)*sin(twopi*A(3))
32    A(4) = ran3(kdum_); A(5) = ran3(kdum_)
      A(6) = A(4)**2*(1.0_real32 - A(4)**2)**3.5_real32
      if (0.1_real32*A(5) > A(6)) goto 32
      A(8) = A(4)*sqrt(2.0_real32)/(1.0_real32 + ri**2)**0.25_real32
      A(6) = ran3(kdum_); A(7) = ran3(kdum_)
      v_(3,p) = (1.0_real32 - 2.0_real32*A(6))*A(8)
      v_(1,p) = sqrt(A(8)**2 - v_(3,p)**2)*cos(twopi*A(7))
      v_(2,p) = sqrt(A(8)**2 - v_(3,p)**2)*sin(twopi*A(7))
    end do
  end subroutine plummer

  ! ============================================================
  ! Mass list
  ! ============================================================
  subroutine write_masses(tag_, N_, m_kg, Msun_)
    character(len=*), intent(in) :: tag_
    integer,  intent(in) :: N_
    real(dp), intent(in) :: m_kg(N_), Msun_
    character(len=128) :: fname
    integer :: u, i
    fname = 'masses_'//trim(tag_)//'.txt'
    u = 21
    open(u, file=fname, status='replace', action='write', form='formatted')
    do i = 1, N_
      write(u,'(F10.6)') m_kg(i)/Msun_
    end do
    close(u)
    write(*,'(A)') 'Mass list: masses_'//trim(tag_)//'.txt'
  end subroutine write_masses

  ! ============================================================
  ! ran3 (Numerical Recipes, unchanged)
  ! ============================================================
  real(real32) function ran3(idum)
    integer, intent(inout) :: idum
    integer, parameter :: MBIG=1000000000, MSEED=161803398, MZ=0
    real(real32), parameter :: FAC=1.0_real32/real(MBIG,real32)
    integer :: i, iff, ii, inext, inextp, k, mj, mk, ma(55)
    save iff, inext, inextp, ma
    data iff /0/

    if (idum < 0 .or. iff == 0) then
      iff = 1
      mj = MSEED - abs(idum); mj = mod(mj, MBIG); ma(55) = mj; mk = 1
      do i = 1, 54
        ii = mod(21*i,55); ma(ii) = mk; mk = mj-mk
        if (mk < MZ) mk = mk+MBIG; mj = ma(ii)
      end do
      do k = 1, 4
        do i = 1, 55
          ma(i) = ma(i) - ma(1+mod(i+30,55))
          if (ma(i) < MZ) ma(i) = ma(i)+MBIG
        end do
      end do
      inext = 0; inextp = 31; idum = 1
    end if

    inext = inext+1; if (inext == 56) inext = 1
    inextp = inextp+1; if (inextp == 56) inextp = 1
    mj = ma(inext)-ma(inextp)
    if (mj < MZ) mj = mj+MBIG
    ma(inext) = mj
    ran3 = real(mj,real32)*FAC
  end function ran3

end program cluster_bhtree_yoshida6