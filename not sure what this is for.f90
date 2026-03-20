! ============================================================================
! cluster_bhtree_leapfrog.f90
! Barnes–Hut octree (monopole) + leapfrog (KDK), OpenMP.
! Designed for large N (1e5).
! Outputs:
!   energy_YYYYMMDD_HHMMSS.txt  : t[s], E[J], dE/E0
!   snaps_YYYYMMDD_HHMMSS.bin   : stream binary snapshots (N32, t, x,y,z in pc as float32)
! ============================================================================

program cluster_bhtree_leapfrog
  use, intrinsic :: iso_fortran_env, only: int32, real32, real64, output_unit
  use omp_lib
  implicit none

  integer, parameter :: dp = real64
  real(dp), parameter :: G    = 6.67430e-11_dp
  real(dp), parameter :: Msun = 1.98847e30_dp
  real(dp), parameter :: pc   = 3.0856775814913673e16_dp
  real(dp), parameter :: Myr  = 1.0e6_dp*365.25_dp*86400.0_dp

  ! ---------------- controls ----------------
  real(dp), parameter :: Rscale_pc = 1.0_dp        ! Plummer scale length a [pc]
  real(dp), parameter :: Q_target  = 1.0_dp        ! virial ratio 2T/|U| (1 = virial)
  real(dp), parameter :: theta     = 0.5_dp        ! BH opening angle
  real(dp), parameter :: eps_m     = 1.0e13_dp     ! softening [m]; set 0 for none

  real(dp), parameter :: t_end_Myr = 2.0_dp        ! total time [Myr]
  real(dp), parameter :: dt_Myr    = 0.002_dp      ! timestep [Myr] (fixed)
  real(dp), parameter :: out_Myr   = 0.02_dp       ! snapshot cadence [Myr]

  integer, parameter :: seed0 = -12345             ! ran3 seed (negative = init)

  ! ---------------- state ----------------
  integer :: N, step, nsteps, write_every, every_print
  real(dp) :: t, dt, t_end, eps2, E0, Ecur, dErel
  real(dp), allocatable :: m(:), r(:,:), v(:,:), a(:,:), anew(:,:)
  integer :: kdum

  ! tree storage
  integer :: max_nodes, node_count
  integer(int32), allocatable :: body(:)
  integer(int32), allocatable :: child(:,:)     ! (8, max_nodes)
  real(dp), allocatable :: mass(:), half(:)
  real(dp), allocatable :: com(:,:), cen(:,:)   ! (3, max_nodes)

  ! output
  character(len=32)  :: tag
  character(len=128) :: energy_name, snaps_name
  integer, parameter :: uener=11, usnap=12
  integer :: ios

  call get_N_from_cli(N)
  if (N <= 0) N = 100000

  dt    = dt_Myr*Myr
  t_end = t_end_Myr*Myr
  eps2  = eps_m*eps_m

  allocate(m(N), r(3,N), v(3,N), a(3,N), anew(3,N))

  max_nodes = 8*N + 16
  allocate(body(max_nodes), child(8,max_nodes), mass(max_nodes), half(max_nodes), com(3,max_nodes), cen(3,max_nodes))

  ! ---------------- initial conditions ----------------
  kdum = seed0
  call make_cluster_ic_plummer(N, Rscale_pc*pc, Q_target, kdum, m, r, v)
  call center_on_com(N, m, r, v)

  ! initial tree + accel
  call build_tree(N, r, max_nodes, node_count, body, child, cen, half)
  call compute_multipoles(N, m, r, node_count, body, child, mass, com)
  call tree_accel_all(N, r, node_count, body, child, mass, com, half, eps2, theta, a)

  E0 = total_energy_tree(N, m, r, v, node_count, body, child, mass, com, half, eps2, theta)

  ! ---------------- output files ----------------
  call timestamp_tag(tag)
  energy_name = 'energy_'//trim(tag)//'.txt'
  snaps_name  = 'snaps_' //trim(tag)//'.bin'

  open(uener, file=energy_name, status='replace', action='write', form='formatted', iostat=ios)
  if (ios /= 0) stop 'Failed to open energy output file.'
  write(uener,'(A)') 't[s]  E[J]  dE/E0'

  open(usnap, file=snaps_name, status='replace', access='stream', form='unformatted', action='write', iostat=ios)
  if (ios /= 0) stop 'Failed to open snapshot output file.'

  ! write initial
  t = 0.0_dp
  call write_energy(uener, t, E0, 0.0_dp)
  call write_snapshot_bin(usnap, N, t, r, pc)

  nsteps = int(t_end/dt)
  write_every = max(1, int((out_Myr*Myr)/dt))
  every_print = max(1, int(0.01_dp*nsteps))

  write(*,'(A,I0)') 'N = ', N
  write(*,'(A,I0)') 'OpenMP threads = ', omp_get_max_threads()
  write(*,'(A,ES12.4)') 'dt [Myr] = ', dt/Myr
  write(*,'(A,ES12.4)') 't_end [Myr] = ', t_end/Myr
  write(*,'(A,ES12.4)') 'theta = ', theta
  write(*,'(A,ES12.4)') 'eps [m] = ', eps_m

  call show_progress(0, nsteps)

  ! ---------------- main loop ----------------
  do step = 1, nsteps

     ! kick
     v = v + 0.5_dp*a*dt

     ! drift
     r = r + v*dt

     ! rebuild tree at new positions
     call build_tree(N, r, max_nodes, node_count, body, child, cen, half)
     call compute_multipoles(N, m, r, node_count, body, child, mass, com)

     ! accel at new positions
     call tree_accel_all(N, r, node_count, body, child, mass, com, half, eps2, theta, anew)

     ! kick
     v = v + 0.5_dp*anew*dt
     a = anew

     t = t + dt

     if (mod(step, write_every) == 0) then
        Ecur  = total_energy_tree(N, m, r, v, node_count, body, child, mass, com, half, eps2, theta)
        dErel = (Ecur - E0)/E0
        call write_energy(uener, t, Ecur, dErel)
        call write_snapshot_bin(usnap, N, t, r, pc)
     end if

     if (mod(step, every_print) == 0) call show_progress(step, nsteps)
  end do

  call show_progress(nsteps, nsteps)
  write(*,*)
  close(uener); close(usnap)

  write(*,'(A)') 'Done.'
  write(*,'(A)') 'Energy file: '//trim(energy_name)
  write(*,'(A)') 'Snaps  file: '//trim(snaps_name)

contains

  subroutine get_N_from_cli(N)
    integer, intent(out) :: N
    character(len=64) :: arg
    integer :: stat
    N = -1
    call get_command_argument(1, arg, status=stat)
    if (stat == 0) then
       read(arg,*,iostat=stat) N
       if (stat /= 0) N = -1
    end if
  end subroutine get_N_from_cli

  ! ---------------- progress bar ----------------
  subroutine show_progress(i, total)
    integer, intent(in) :: i, total
    integer, parameter  :: width = 50
    real(dp) :: frac
    integer  :: filled
    character(len=width) :: bar
    frac   = real(i,dp)/max(1,total)
    filled = int(frac*width)
    if (filled < 0) filled = 0
    if (filled > width) filled = width
    bar = repeat('#', filled)//repeat('-', width - filled)
    write(output_unit,'(A)',advance='no') achar(13)//'['//bar//'] ' // trim(adjustl(pct_str(100.0_dp*frac))) // '%'
    call flush(output_unit)
  end subroutine show_progress

  pure function pct_str(x) result(s)
    real(dp), intent(in) :: x
    character(len=6)     :: s
    write(s,'(F5.1)') x
  end function pct_str

  subroutine write_energy(u, t, E, dErel)
    integer, intent(in) :: u
    real(dp), intent(in) :: t, E, dErel
    write(u,'(ES25.16,2(1X,ES25.16))') t, E, dErel
  end subroutine write_energy

  subroutine write_snapshot_bin(u, N, t, r, pc_unit)
    integer, intent(in) :: u, N
    real(dp), intent(in) :: t, r(3,N), pc_unit
    integer(int32) :: N32
    real(real32), allocatable, save :: x(:), y(:), z(:)
    integer, save :: nalloc = 0
    integer :: i

    if (nalloc /= N) then
      if (allocated(x)) deallocate(x,y,z)
      allocate(x(N), y(N), z(N))
      nalloc = N
    end if

    do i = 1, N
      x(i) = real(r(1,i)/pc_unit, real32)
      y(i) = real(r(2,i)/pc_unit, real32)
      z(i) = real(r(3,i)/pc_unit, real32)
    end do

    N32 = int(N, int32)
    write(u) N32
    write(u) t
    write(u) x
    write(u) y
    write(u) z
  end subroutine write_snapshot_bin

  ! ---------------- cluster ICs ----------------
  subroutine make_cluster_ic_plummer(N, Rscale_m, Q, kdum, m, r, v)
    integer, intent(in) :: N
    real(dp), intent(in) :: Rscale_m, Q
    integer, intent(inout) :: kdum
    real(dp), intent(out) :: m(N), r(3,N), v(3,N)

    integer :: i
    real(real32), allocatable :: r0(:,:), v0(:,:)
    real(dp) :: T, U, scale

    allocate(r0(3,N), v0(3,N))

    ! masses (kg)
    do i = 1, N
       m(i) = sample_kroupa_02_4(kdum)*Msun
    end do

    ! Plummer positions/velocities in dimensionless N-body units
    call plummer(N, r0, v0, kdum)

    ! scale to physical length; velocities become physical after virial scaling
    do i = 1, N
       r(:,i) = real(r0(:,i),dp) * Rscale_m
       v(:,i) = real(v0(:,i),dp)
    end do

    call center_on_com(N, m, r, v)

    ! Scale velocities to desired virial ratio Q = 2T/|U|
    call kinetic_potential_pairwise(N, m, r, v, eps2=0.0_dp, T=T, U=U)
    if (T <= 0.0_dp .or. U >= 0.0_dp) stop 'Bad initial energies (check IC generator).'
    scale = sqrt((Q*abs(U)) / (2.0_dp*T))
    v = v * scale

    call center_on_com(N, m, r, v)

    deallocate(r0, v0)
  end subroutine make_cluster_ic_plummer

  ! Kroupa-like IMF (0.2–4 Msun)
  real(dp) function sample_kroupa_02_4(kdum) result(mass_msun)
    integer, intent(inout) :: kdum
    real(dp), parameter :: m1=0.2_dp, mb=0.5_dp, m2=4.0_dp
    real(dp), parameter :: a1=1.3_dp, a2=2.3_dp
    real(dp) :: u, w1, w2
    w1 = integral_powerlaw(m1, mb, a1)
    w2 = integral_powerlaw(mb, m2, a2)
    u  = ran_u(kdum)
    if (u < w1/(w1+w2)) then
       mass_msun = sample_powerlaw(m1, mb, a1, kdum)
    else
       mass_msun = sample_powerlaw(mb, m2, a2, kdum)
    end if
  end function sample_kroupa_02_4

  real(dp) function integral_powerlaw(mmin, mmax, alpha) result(val)
    real(dp), intent(in) :: mmin, mmax, alpha
    if (abs(alpha-1.0_dp) < 1.0e-12_dp) then
       val = log(mmax/mmin)
    else
       val = (mmax**(1.0_dp-alpha) - mmin**(1.0_dp-alpha)) / (1.0_dp-alpha)
    end if
  end function integral_powerlaw

  real(dp) function sample_powerlaw(mmin, mmax, alpha, kdum) result(mass)
    real(dp), intent(in) :: mmin, mmax, alpha
    integer, intent(inout) :: kdum
    real(dp) :: u
    u = ran_u(kdum)
    if (abs(alpha-1.0_dp) < 1.0e-12_dp) then
       mass = mmin * (mmax/mmin)**u
    else
       mass = ( u*(mmax**(1.0_dp-alpha) - mmin**(1.0_dp-alpha)) + mmin**(1.0_dp-alpha) )**(1.0_dp/(1.0_dp-alpha))
    end if
  end function sample_powerlaw

  real(dp) function ran_u(kdum) result(u)
    integer, intent(inout) :: kdum
    real(real32) :: x
    x = ran3(kdum)
    if (x <= 0.0_real32) x = 1.0e-7_real32
    if (x >= 1.0_real32) x = 1.0_real32 - 1.0e-7_real32
    u = real(x, dp)
  end function ran_u

  subroutine center_on_com(N, m, r, v)
    integer, intent(in)    :: N
    real(dp), intent(in)   :: m(N)
    real(dp), intent(inout):: r(3,N), v(3,N)
    integer  :: i
    real(dp) :: rcom(3), vcom(3), Mtot
    Mtot = sum(m)
    rcom = 0.0_dp; vcom = 0.0_dp
    do i = 1, N
       rcom = rcom + m(i)*r(:,i)
       vcom = vcom + m(i)*v(:,i)
    end do
    rcom = rcom/Mtot; vcom = vcom/Mtot
    do i = 1, N
       r(:,i) = r(:,i) - rcom
       v(:,i) = v(:,i) - vcom
    end do
  end subroutine center_on_com

  ! ---------------- tree build ----------------
  subroutine build_tree(N, r, max_nodes, node_count, body, child, cen, half)
    integer, intent(in) :: N, max_nodes
    real(dp), intent(in) :: r(3,N)
    integer, intent(out) :: node_count
    integer(int32), intent(inout) :: body(max_nodes)
    integer(int32), intent(inout) :: child(8,max_nodes)
    real(dp), intent(inout) :: cen(3,max_nodes), half(max_nodes)

    integer :: i
    real(dp) :: rmin(3), rmax(3), c(3), h

    rmin = r(:,1)
    rmax = r(:,1)
    do i = 2, N
      rmin = min(rmin, r(:,i))
      rmax = max(rmax, r(:,i))
    end do

    c = 0.5_dp*(rmin + rmax)
    h = 0.5_dp*maxval(rmax - rmin)
    if (h <= 0.0_dp) h = 1.0_dp
    h = 1.001_dp*h

    node_count = 1
    body(1) = 0
    child(:,1) = 0
    cen(:,1) = c
    half(1) = h

    do i = 1, N
      call insert_body(1, i, r, max_nodes, node_count, body, child, cen, half)
    end do
  end subroutine build_tree

  subroutine insert_body(root, ibody, r, max_nodes, node_count, body, child, cen, half)
    integer, intent(in) :: root, ibody, max_nodes
    real(dp), intent(in) :: r(3,*)
    integer, intent(inout) :: node_count
    integer(int32), intent(inout) :: body(max_nodes)
    integer(int32), intent(inout) :: child(8,max_nodes)
    real(dp), intent(inout) :: cen(3,max_nodes), half(max_nodes)

    integer :: node, oct, c, old

    node = root
    do
      if (body(node) == 0 .and. all(child(:,node) == 0)) then
         body(node) = int(ibody, int32)
         return

      else if (all(child(:,node) == 0) .and. body(node) > 0) then
         old = int(body(node))
         body(node) = -1
         call create_children(node, max_nodes, node_count, body, child, cen, half)
         call insert_body_into_child(node, old, r, max_nodes, node_count, body, child, cen, half)
         call insert_body_into_child(node, ibody, r, max_nodes, node_count, body, child, cen, half)
         return

      else
         oct = octant(cen(:,node), r(:,ibody))
         c = int(child(oct,node))
         node = c
      end if
    end do
  end subroutine insert_body

  subroutine insert_body_into_child(node, ibody, r, max_nodes, node_count, body, child, cen, half)
    integer, intent(in) :: node, ibody, max_nodes
    real(dp), intent(in) :: r(3,*)
    integer, intent(inout) :: node_count
    integer(int32), intent(inout) :: body(max_nodes)
    integer(int32), intent(inout) :: child(8,max_nodes)
    real(dp), intent(inout) :: cen(3,max_nodes), half(max_nodes)
    integer :: oct, c

    oct = octant(cen(:,node), r(:,ibody))
    c = int(child(oct,node))
    call insert_body(c, ibody, r, max_nodes, node_count, body, child, cen, half)
  end subroutine insert_body_into_child

  subroutine create_children(node, max_nodes, node_count, body, child, cen, half)
    integer, intent(in) :: node, max_nodes
    integer, intent(inout) :: node_count
    integer(int32), intent(inout) :: body(max_nodes)
    integer(int32), intent(inout) :: child(8,max_nodes)
    real(dp), intent(inout) :: cen(3,max_nodes), half(max_nodes)

    integer :: o, c
    real(dp) :: ch, dx, dy, dz

    ch = 0.5_dp*half(node)
    do o = 1, 8
      node_count = node_count + 1
      if (node_count > max_nodes) stop 'Tree overflow: increase max_nodes.'
      c = node_count
      child(o,node) = int(c, int32)

      body(c) = 0
      child(:,c) = 0

      dx = merge(ch, -ch, btest(o-1,0))
      dy = merge(ch, -ch, btest(o-1,1))
      dz = merge(ch, -ch, btest(o-1,2))
      cen(:,c) = cen(:,node) + [dx, dy, dz]
      half(c)  = ch
    end do
  end subroutine create_children

  integer function octant(c, x) result(o)
    real(dp), intent(in) :: c(3), x(3)
    integer :: bx, by, bz
    bx = 0; by = 0; bz = 0
    if (x(1) >= c(1)) bx = 1
    if (x(2) >= c(2)) by = 1
    if (x(3) >= c(3)) bz = 1
    o = 1 + bx + 2*by + 4*bz
  end function octant

  subroutine compute_multipoles(N, m, r, node_count, body, child, mass, com)
    integer, intent(in) :: N, node_count
    real(dp), intent(in) :: m(N), r(3,N)
    integer(int32), intent(in) :: body(node_count), child(8,node_count)
    real(dp), intent(out) :: mass(node_count), com(3,node_count)

    integer :: n, o, c, b
    real(dp) :: mt

    mass = 0.0_dp
    com  = 0.0_dp

    do n = node_count, 1, -1
      if (body(n) > 0) then
        b = int(body(n))
        mass(n) = m(b)
        com(:,n) = r(:,b)
      else
        mt = 0.0_dp
        com(:,n) = 0.0_dp
        do o = 1, 8
          c = int(child(o,n))
          if (c <= 0) cycle
          mt = mt + mass(c)
          com(:,n) = com(:,n) + mass(c)*com(:,c)
        end do
        if (mt > 0.0_dp) then
          com(:,n) = com(:,n)/mt
          mass(n) = mt
        else
          mass(n) = 0.0_dp
          com(:,n) = 0.0_dp
        end if
      end if
    end do
  end subroutine compute_multipoles

  ! ---------------- tree force (OpenMP) ----------------
  subroutine tree_accel_all(N, r, node_count, body, child, mass, com, half, eps2, theta, a)
    integer, intent(in) :: N, node_count
    real(dp), intent(in) :: r(3,N), mass(node_count), com(3,node_count), half(node_count)
    integer(int32), intent(in) :: body(node_count), child(8,node_count)
    real(dp), intent(in) :: eps2, theta
    real(dp), intent(out) :: a(3,N)

    integer, parameter :: max_stack = 1024
    integer :: i, top, n, o, c, b
    integer :: stack(max_stack)
    real(dp) :: dx(3), r2, invr, invr3, s2, theta2

    theta2 = theta*theta

    !$omp parallel do default(none) shared(N,r,node_count,body,child,mass,com,half,eps2,theta2,a) &
    !$omp private(i,top,n,o,c,b,stack,dx,r2,invr,invr3,s2) schedule(static)
    do i = 1, N
      a(:,i) = 0.0_dp
      top = 1
      stack(1) = 1

      do while (top > 0)
        n = stack(top); top = top - 1
        if (mass(n) == 0.0_dp) cycle

        if (body(n) > 0) then
          b = int(body(n))
          if (b == i) cycle
          dx = com(:,n) - r(:,i)
          r2 = dx(1)*dx(1) + dx(2)*dx(2) + dx(3)*dx(3) + eps2
          invr = 1.0_dp/sqrt(r2)
          invr3 = invr*invr*invr
          a(:,i) = a(:,i) + G*mass(n)*dx*invr3
        else
          dx = com(:,n) - r(:,i)
          r2 = dx(1)*dx(1) + dx(2)*dx(2) + dx(3)*dx(3) + eps2
          s2 = (2.0_dp*half(n))**2

          if (s2 < theta2*r2) then
            invr = 1.0_dp/sqrt(r2)
            invr3 = invr*invr*invr
            a(:,i) = a(:,i) + G*mass(n)*dx*invr3
          else
            do o = 1, 8
              c = int(child(o,n))
              if (c <= 0) cycle
              if (top >= max_stack) stop 'Tree stack overflow (increase max_stack).'
              top = top + 1
              stack(top) = c
            end do
          end if
        end if
      end do
    end do
    !$omp end parallel do
  end subroutine tree_accel_all

  subroutine tree_potential_all(N, r, node_count, body, child, mass, com, half, eps2, theta, phi)
    integer, intent(in) :: N, node_count
    real(dp), intent(in) :: r(3,N), mass(node_count), com(3,node_count), half(node_count)
    integer(int32), intent(in) :: body(node_count), child(8,node_count)
    real(dp), intent(in) :: eps2, theta
    real(dp), intent(out) :: phi(N)

    integer, parameter :: max_stack = 1024
    integer :: i, top, n, o, c, b
    integer :: stack(max_stack)
    real(dp) :: dx(3), r2, invr, s2, theta2

    theta2 = theta*theta

    !$omp parallel do default(none) shared(N,r,node_count,body,child,mass,com,half,eps2,theta2,phi) &
    !$omp private(i,top,n,o,c,b,stack,dx,r2,invr,s2) schedule(static)
    do i = 1, N
      phi(i) = 0.0_dp
      top = 1
      stack(1) = 1

      do while (top > 0)
        n = stack(top); top = top - 1
        if (mass(n) == 0.0_dp) cycle

        if (body(n) > 0) then
          b = int(body(n))
          if (b == i) cycle
          dx = com(:,n) - r(:,i)
          r2 = dx(1)*dx(1) + dx(2)*dx(2) + dx(3)*dx(3) + eps2
          invr = 1.0_dp/sqrt(r2)
          phi(i) = phi(i) - G*mass(n)*invr
        else
          dx = com(:,n) - r(:,i)
          r2 = dx(1)*dx(1) + dx(2)*dx(2) + dx(3)*dx(3) + eps2
          s2 = (2.0_dp*half(n))**2

          if (s2 < theta2*r2) then
            invr = 1.0_dp/sqrt(r2)
            phi(i) = phi(i) - G*mass(n)*invr
          else
            do o = 1, 8
              c = int(child(o,n))
              if (c <= 0) cycle
              if (top >= max_stack) stop 'Tree stack overflow (increase max_stack).'
              top = top + 1
              stack(top) = c
            end do
          end if
        end if
      end do
    end do
    !$omp end parallel do
  end subroutine tree_potential_all

  ! ---------------- energy ----------------
  real(dp) function total_energy_tree(N, m, r, v, node_count, body, child, mass, com, half, eps2, theta) result(E)
    integer, intent(in) :: N, node_count
    real(dp), intent(in) :: m(N), r(3,N), v(3,N), mass(node_count), com(3,node_count), half(node_count)
    integer(int32), intent(in) :: body(node_count), child(8,node_count)
    real(dp), intent(in) :: eps2, theta

    real(dp) :: T, U
    real(dp), allocatable, save :: phi(:)
    integer, save :: nphi = 0
    integer :: i

    T = 0.0_dp
    do i = 1, N
      T = T + 0.5_dp*m(i)*( v(1,i)*v(1,i) + v(2,i)*v(2,i) + v(3,i)*v(3,i) )
    end do

    if (nphi /= N) then
      if (allocated(phi)) deallocate(phi)
      allocate(phi(N))
      nphi = N
    end if

    call tree_potential_all(N, r, node_count, body, child, mass, com, half, eps2, theta, phi)
    U = 0.0_dp
    do i = 1, N
      U = U + 0.5_dp*m(i)*phi(i)
    end do

    E = T + U
  end function total_energy_tree

  subroutine kinetic_potential_pairwise(N, m, r, v, eps2, T, U)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N), r(3,N), v(3,N), eps2
    real(dp), intent(out) :: T, U
    integer :: i, k
    real(dp) :: dx(3), r2, invr

    T = 0.0_dp
    do i = 1, N
      T = T + 0.5_dp*m(i)*( v(1,i)*v(1,i) + v(2,i)*v(2,i) + v(3,i)*v(3,i) )
    end do

    U = 0.0_dp
    do i = 1, N-1
      do k = i+1, N
        dx = r(:,k) - r(:,i)
        r2 = dx(1)*dx(1) + dx(2)*dx(2) + dx(3)*dx(3) + eps2
        invr = 1.0_dp/sqrt(r2)
        U = U - G*m(i)*m(k)*invr
      end do
    end do
  end subroutine kinetic_potential_pairwise

  subroutine timestamp_tag(tag)
    character(len=*), intent(out) :: tag
    integer :: v(8)
    call date_and_time(values=v)
    write(tag,'(I4.4,I2.2,I2.2,"_",I2.2,I2.2,I2.2)') v(1),v(2),v(3),v(5),v(6),v(7)
  end subroutine timestamp_tag

  ! ---------------- provided routines (plummer + ran3) ----------------
  subroutine plummer(n,r,v,kdum)
    implicit none
    integer :: p
    integer, intent(inout) :: kdum
    integer, intent(in) :: n
    real(real32), intent(out) :: r(1:3,1:n),v(1:3,1:n)
    real(real32) :: A(8),ri,twopi

    twopi = 8.0_real32*atan(1.0_real32)

    do p = 1, n
30     A(1)=ran3(kdum)
       if (A(1)<1.0e-10_real32) goto 30
       ri = (A(1)**(-0.6666667_real32) - 1.0_real32)**(-0.5_real32)
       if (ri>14.6154_real32) goto 30

       A(2)=ran3(kdum)
       A(3)=ran3(kdum)
       r(3,p)=(1.0_real32 - 2.0_real32*A(2))*ri
       r(1,p)=sqrt(ri**2 - r(3,p)**2)*cos(twopi*A(3))
       r(2,p)=sqrt(ri**2 - r(3,p)**2)*sin(twopi*A(3))

32     A(4)=ran3(kdum)
       A(5)=ran3(kdum)
       A(6)=A(4)**2*(1.0_real32 - A(4)**2)**3.5_real32
       if (0.1_real32*A(5)>A(6)) goto 32

       A(8) = A(4)*sqrt(2.0_real32)/(1.0_real32 + ri**2)**0.25_real32
       A(6) = ran3(kdum)
       A(7) = ran3(kdum)
       v(3,p) = (1.0_real32 - 2.0_real32*A(6))*A(8)
       v(1,p) = sqrt(A(8)**2 - v(3,p)**2)*cos(twopi*A(7))
       v(2,p) = sqrt(A(8)**2 - v(3,p)**2)*sin(twopi*A(7))
    end do
  end subroutine plummer

  real(real32) function ran3(idum)
    implicit none
    integer, intent(inout) :: idum
    integer, parameter :: MBIG=1000000000, MSEED=161803398, MZ=0
    real(real32), parameter :: FAC=1.0_real32/real(MBIG,real32)
    integer :: i, iff, ii, inext, inextp, k
    integer :: mj, mk
    integer :: ma(55)
    save iff, inext, inextp, ma
    data iff /0/

    if (idum < 0 .or. iff == 0) then
      iff = 1
      mj = MSEED - abs(idum)
      mj = mod(mj, MBIG)
      ma(55) = mj
      mk = 1
      do i = 1, 54
        ii = mod(21*i,55)
        ma(ii) = mk
        mk = mj - mk
        if (mk < MZ) mk = mk + MBIG
        mj = ma(ii)
      end do
      do k = 1, 4
        do i = 1, 55
          ma(i) = ma(i) - ma(1 + mod(i+30,55))
          if (ma(i) < MZ) ma(i) = ma(i) + MBIG
        end do
      end do
      inext = 0
      inextp = 31
      idum = 1
    end if

    inext = inext + 1
    if (inext == 56) inext = 1
    inextp = inextp + 1
    if (inextp == 56) inextp = 1
    mj = ma(inext) - ma(inextp)
    if (mj < MZ) mj = mj + MBIG
    ma(inext) = mj
    ran3 = real(mj,real32)*FAC
  end function ran3

end program cluster_bhtree_leapfrog
