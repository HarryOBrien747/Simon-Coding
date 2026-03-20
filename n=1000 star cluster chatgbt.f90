! nbody_cluster_hermite4_block.f90
! 4th-order Hermite N-body with block (power-of-two) individual timesteps + OpenMP
!
! Default: N=1000 star cluster
! - Masses: Kroupa-like broken power law, 0.2–4 Msun (high masses rarer)
! - Positions: uniform sphere (Rcluster)
! - Velocities: Gaussian ~1 km/s then scaled to virial equilibrium (Q = 2T/|U| = 1)
! - Integration: 4th order Hermite with individual block timesteps
!
! Compile (example):
!   gfortran -O3 -march=native -ffast-math -fopenmp nbody_cluster_hermite4_block.f90 -o cluster
!
! Run:
!   ./cluster
!   (optionally set threads: export OMP_NUM_THREADS=8)

program nbody_cluster_hermite4_block
  use, intrinsic :: iso_fortran_env, only: output_unit
  use omp_lib
  implicit none

  integer, parameter :: dp = selected_real_kind(15, 300)
  integer, parameter :: i8 = selected_int_kind(18)

  ! -------------------- physical constants --------------------
  real(dp), parameter :: G    = 6.67430e-11_dp
  real(dp), parameter :: Msun = 1.98847e30_dp
  real(dp), parameter :: pc   = 3.085677581e16_dp
  real(dp), parameter :: Myr  = 1.0e6_dp * (365.25_dp*86400.0_dp)

  ! -------------------- problem size --------------------
  integer, parameter :: N = 1000

  ! -------------------- simulation controls --------------------
  real(dp), parameter :: eta     = 0.02_dp         ! timestep accuracy (typ. 0.01–0.03)
  real(dp), parameter :: dt_min  = 67108864.0_dp   ! base step [s] = 2^26 s ≈ 2.13 yr
  integer,  parameter :: max_pow = 16              ! dt_max = dt_min * 2^max_pow
  real(dp), parameter :: t_end   = 2.0_dp*Myr      ! total time [s] (short test; increase later)

  ! Initial condition controls
  real(dp), parameter :: Rcluster = 1.0_dp*pc      ! initial radius [m]
  real(dp), parameter :: vrms0    = 1.0e3_dp       ! ~1 km/s RMS before virial scaling
  real(dp), parameter :: Q_target = 1.0_dp         ! Q=2T/|U| ; Q=1 is virialised

  ! Optional gravitational softening (helps avoid singularities in dense clusters)
  ! If you want it, try eps = 1.0e-4*pc or smaller.
  real(dp), parameter :: eps      = 0.0_dp         ! [m]
  real(dp), parameter :: eps2     = eps*eps

  ! Output controls
  real(dp), parameter :: out_dt   = 0.01_dp*Myr    ! snapshot cadence [s]

  ! -------------------- state arrays --------------------
  real(dp) :: m(N)
  real(dp) :: r(3,N), v(3,N), a(3,N), j(3,N)

  ! Predicted state at current global time
  real(dp) :: rpred(3,N), vpred(3,N)

  ! Time bookkeeping in integer ticks of dt_min
  integer(i8) :: it, it_end
  integer(i8) :: it_last(N), dt_step(N), it_next(N)

  ! Active list
  integer :: active_idx(N)
  integer :: nactive

  ! I/O
  integer, parameter :: ustate=10, uener=11
  integer :: ios
  character(len=32)  :: tag
  character(len=128) :: state_name, energy_name

  ! Energy
  real(dp) :: E0, Ecur, dErel

  ! Output timing
  integer(i8) :: it_out_next, out_stride

  integer(i8) :: step_print

  ! -------------------- init --------------------
  call seed_rng()
  call make_cluster_ic(N, Rcluster, vrms0, Q_target, m, r, v)

  ! Initial accelerations/jerks at t=0
  call acc_jerk_all(N, m, r, v, a, j)

  ! Initialise block timestep schedule
  call init_timesteps(N, eta, dt_min, max_pow, a, j, dt_step)
  it_last = 0_i8
  it_next = it_last + dt_step

  it_end = int(t_end/dt_min, kind=i8)

  out_stride  = max(1_i8, int(out_dt/dt_min, kind=i8))
  it_out_next = out_stride

  ! Open output files
  call timestamp_tag(tag)
  state_name  = 'cluster_state_'  // trim(tag) // '.txt'
  energy_name = 'cluster_energy_' // trim(tag) // '.txt'

  open(ustate, file=state_name, status='replace', action='write', form='formatted', iostat=ios)
  if (ios /= 0) stop 'Failed to open state output file.'
  call write_state_header(ustate, N)

  open(uener, file=energy_name, status='replace', action='write', form='formatted', iostat=ios)
  if (ios /= 0) stop 'Failed to open energy output file.'
  write(uener,'(A)') 't[s] E[J] dE/E0'

  ! Reference energy at t=0 (predicted == actual at t=0)
  call predict_all(N, 0_i8, it_last, dt_min, r, v, a, j, rpred, vpred)
  E0 = total_energy(N, m, rpred, vpred)

  call dump_state(ustate, N, 0.0_dp, rpred, vpred)
  write(uener,'(ES25.16,2(1X,ES25.16))') 0.0_dp, E0, 0.0_dp

  print *, 'N =', N
  print *, 'OpenMP threads =', omp_get_max_threads()
  print *, 'dt_min [yr] =', dt_min/(365.25_dp*86400.0_dp)
  print *, 't_end [Myr] =', t_end/Myr
  if (eps > 0.0_dp) print *, 'Softening eps [pc] =', eps/pc

  step_print = max(1_i8, it_end/100_i8)   ! ~100 progress updates

  ! -------------------- main loop: global tick it --------------------
  do it = 1_i8, it_end

     ! 1) Predict r,v for ALL particles at current global time it
     call predict_all(N, it, it_last, dt_min, r, v, a, j, rpred, vpred)

     ! 2) Determine active particles (whose next update is now)
     call build_active_list(N, it, it_next, nactive, active_idx)

     ! 3) Update active particles using Hermite corrector with their own dt
     if (nactive > 0) then
        call advance_active(N, nactive, active_idx, m, it, it_last, dt_step, dt_min, max_pow, eta, &
                            r, v, a, j, rpred, vpred, it_next)
     end if

     ! 4) Output on schedule (use predicted state at this global time)
     if (it == it_out_next) then
        call predict_all(N, it, it_last, dt_min, r, v, a, j, rpred, vpred)
        Ecur  = total_energy(N, m, rpred, vpred)
        dErel = (Ecur - E0)/E0
        call dump_state(ustate, N, real(it,dp)*dt_min, rpred, vpred)
        write(uener,'(ES25.16,2(1X,ES25.16))') real(it,dp)*dt_min, Ecur, dErel
        it_out_next = it_out_next + out_stride
     end if

     ! 5) Progress bar
     if (mod(it, step_print) == 0_i8) call show_progress(it, it_end)

  end do

  call show_progress(it_end, it_end)
  write(*,*)
  close(ustate); close(uener)

  print *, 'Wrote:', trim(state_name)
  print *, 'Wrote:', trim(energy_name)

contains

  subroutine seed_rng()
    integer :: n, k, count
    integer, allocatable :: seed(:)
    call random_seed(size=n)
    allocate(seed(n))
    call system_clock(count=count)
    do k = 1, n
       seed(k) = count + 37*k
    end do
    call random_seed(put=seed)
    deallocate(seed)
  end subroutine seed_rng

  subroutine timestamp_tag(tag)
    character(len=*), intent(out) :: tag
    integer :: v(8)
    call date_and_time(values=v)
    write(tag,'(I4.4,I2.2,I2.2,"_",I2.2,I2.2,I2.2)') v(1),v(2),v(3),v(5),v(6),v(7)
  end subroutine timestamp_tag

  subroutine show_progress(i, total)
    integer(i8), intent(in) :: i, total
    integer, parameter  :: width = 50
    real(dp) :: frac
    integer  :: filled
    character(len=width) :: bar
    frac   = real(i,dp)/max(1.0_dp, real(total,dp))
    filled = int(frac*width)
    filled = max(0, min(width, filled))
    bar = repeat('#', filled)//repeat('-', width - filled)
    write(output_unit,'(A)',advance='no') achar(13)//'['//trim(bar)//'] ' // trim(pct_str(100.0_dp*frac)) // '%'
    call flush(output_unit)
  end subroutine show_progress

  pure function pct_str(x) result(s)
    real(dp), intent(in) :: x
    character(len=6)     :: s
    write(s,'(F5.1)') x
  end function pct_str

  ! -------------------- initial conditions --------------------
  subroutine make_cluster_ic(N, Rcl, vrms, Q, m, r, v)
    integer, intent(in) :: N
    real(dp), intent(in) :: Rcl, vrms, Q
    real(dp), intent(out) :: m(N), r(3,N), v(3,N)
    integer :: i
    real(dp) :: T, U, scale, sigma

    ! 1) masses
    do i = 1, N
       m(i) = sample_mass_msun() * Msun
    end do

    ! 2) positions: uniform in a sphere of radius R
    do i = 1, N
       call sample_uniform_sphere(Rcl, r(:,i))
    end do

    ! 3) velocities: 3D Gaussian, roughly vrms overall (before virial scaling)
    sigma = vrms / sqrt(3.0_dp)
    do i = 1, N
       call gaussian3(sigma, v(:,i))
    end do

    ! 4) remove COM
    call center_on_com(N, m, r, v)

    ! 5) virial scale velocities to target Q=2T/|U|
    call kinetic_potential(N, m, r, v, T, U)
    if (T <= 0.0_dp .or. U >= 0.0_dp) stop 'Bad initial energies (check IC generator).'
    scale = sqrt((Q*abs(U)) / (2.0_dp*T))
    v = v * scale

    ! remove COM velocity again
    call center_on_com(N, m, r, v)
  end subroutine make_cluster_ic

  real(dp) function sample_mass_msun()
    ! Broken power-law IMF (Kroupa-like) in Msun: [0.2,0.5] alpha=1.3; [0.5,4] alpha=2.3
    real(dp), parameter :: m1=0.2_dp, mb=0.5_dp, m2=4.0_dp
    real(dp), parameter :: a1=1.3_dp, a2=2.3_dp
    real(dp) :: u, w1, w2

    call random_number(u)

    w1 = integral_pl(m1, mb, a1)
    w2 = integral_pl(mb, m2, a2)

    if (u < w1/(w1+w2)) then
       call random_number(u)
       sample_mass_msun = inv_cdf_pl(m1, mb, a1, u)
    else
       call random_number(u)
       sample_mass_msun = inv_cdf_pl(mb, m2, a2, u)
    end if
  end function sample_mass_msun

  pure real(dp) function integral_pl(mlo, mhi, alpha)
    real(dp), intent(in) :: mlo, mhi, alpha
    if (abs(alpha-1.0_dp) < 1e-12_dp) then
       integral_pl = log(mhi/mlo)
    else
       integral_pl = (mhi**(1.0_dp-alpha) - mlo**(1.0_dp-alpha)) / (1.0_dp-alpha)
    end if
  end function integral_pl

  pure real(dp) function inv_cdf_pl(mlo, mhi, alpha, u)
    real(dp), intent(in) :: mlo, mhi, alpha, u
    real(dp) :: A
    if (abs(alpha-1.0_dp) < 1e-12_dp) then
       inv_cdf_pl = mlo * exp(u * log(mhi/mlo))
    else
       A = mlo**(1.0_dp-alpha) + u*(mhi**(1.0_dp-alpha) - mlo**(1.0_dp-alpha))
       inv_cdf_pl = A**(1.0_dp/(1.0_dp-alpha))
    end if
  end function inv_cdf_pl

  subroutine sample_uniform_sphere(R, x)
    real(dp), intent(in) :: R
    real(dp), intent(out) :: x(3)
    real(dp) :: u1, u2, u3, rr, costh, sinth, phi
    call random_number(u1)
    call random_number(u2)
    call random_number(u3)

    rr    = R * u1**(1.0_dp/3.0_dp)
    costh = 2.0_dp*u2 - 1.0_dp
    sinth = sqrt(max(0.0_dp, 1.0_dp - costh*costh))
    phi   = 2.0_dp*acos(-1.0_dp)*u3

    x(1) = rr * sinth * cos(phi)
    x(2) = rr * sinth * sin(phi)
    x(3) = rr * costh
  end subroutine sample_uniform_sphere

  subroutine gaussian3(sigma, vv)
    real(dp), intent(in) :: sigma
    real(dp), intent(out) :: vv(3)
    vv(1) = sigma * gaussian01()
    vv(2) = sigma * gaussian01()
    vv(3) = sigma * gaussian01()
  end subroutine gaussian3

  real(dp) function gaussian01()
    ! Box–Muller transform, mean 0, std 1
    real(dp) :: u1, u2
    call random_number(u1)
    call random_number(u2)
    u1 = max(u1, 1.0e-12_dp)
    gaussian01 = sqrt(-2.0_dp*log(u1)) * cos(2.0_dp*acos(-1.0_dp)*u2)
  end function gaussian01

  ! -------------------- prediction --------------------
  subroutine predict_all(N, it, it_last, dt_min, r, v, a, j, rpred, vpred)
    integer, intent(in) :: N
    integer(i8), intent(in) :: it
    integer(i8), intent(in) :: it_last(N)
    real(dp), intent(in) :: dt_min
    real(dp), intent(in) :: r(3,N), v(3,N), a(3,N), j(3,N)
    real(dp), intent(out) :: rpred(3,N), vpred(3,N)
    integer :: k
    real(dp) :: dt, dt2, dt3

    !$omp parallel do default(none) shared(N,it,it_last,dt_min,r,v,a,j,rpred,vpred) private(k,dt,dt2,dt3) schedule(static)
    do k = 1, N
       dt  = real(it - it_last(k), dp) * dt_min
       dt2 = dt*dt
       dt3 = dt2*dt
       rpred(:,k) = r(:,k) + v(:,k)*dt + 0.5_dp*a(:,k)*dt2 + (1.0_dp/6.0_dp)*j(:,k)*dt3
       vpred(:,k) = v(:,k) + a(:,k)*dt + 0.5_dp*j(:,k)*dt2
    end do
    !$omp end parallel do
  end subroutine predict_all

  ! -------------------- active particles --------------------
  subroutine build_active_list(N, it, it_next, nactive, active_idx)
    integer, intent(in) :: N
    integer(i8), intent(in) :: it
    integer(i8), intent(in) :: it_next(N)
    integer, intent(out) :: nactive
    integer, intent(out) :: active_idx(N)
    integer :: i

    nactive = 0
    do i = 1, N
       if (it_next(i) == it) then
          nactive = nactive + 1
          active_idx(nactive) = i
       end if
    end do
  end subroutine build_active_list

  subroutine advance_active(N, nactive, active_idx, m, it, it_last, dt_step, dt_min, max_pow, eta, &
                            r, v, a, j, rpred, vpred, it_next)
    integer, intent(in) :: N, nactive, max_pow
    integer, intent(in) :: active_idx(nactive)
    real(dp), intent(in) :: m(N), dt_min, eta
    integer(i8), intent(in) :: it
    integer(i8), intent(inout) :: it_last(N), dt_step(N), it_next(N)
    real(dp), intent(inout) :: r(3,N), v(3,N), a(3,N), j(3,N)
    real(dp), intent(in) :: rpred(3,N), vpred(3,N)

    integer :: ia, i
    real(dp) :: dt, dt2
    real(dp) :: r0(3), v0(3), a0(3), j0(3)
    real(dp) :: ap(3), jp(3)
    real(dp) :: vnew(3), rnew(3)
    real(dp) :: a2(3), a3(3)
    real(dp) :: an, jn, a2n, a3n
    real(dp) :: dt_new
    integer(i8) :: new_step

    !$omp parallel do default(none) shared(N,nactive,active_idx,m,it,it_last,dt_step,dt_min,max_pow,eta,r,v,a,j,rpred,vpred,it_next) &
    !$omp private(ia,i,dt,dt2,r0,v0,a0,j0,ap,jp,vnew,rnew,a2,a3,an,jn,a2n,a3n,dt_new,new_step) schedule(static)
    do ia = 1, nactive
       i = active_idx(ia)

       dt  = real(it - it_last(i), dp) * dt_min
       dt2 = dt*dt

       r0 = r(:,i)
       v0 = v(:,i)
       a0 = a(:,i)
       j0 = j(:,i)

       call acc_jerk_one(N, i, m, rpred, vpred, ap, jp)

       ! Hermite corrector (4th order)
       vnew = v0 + 0.5_dp*(a0 + ap)*dt + (1.0_dp/12.0_dp)*(j0 - jp)*dt2
       rnew = r0 + 0.5_dp*(v0 + vnew)*dt + (1.0_dp/12.0_dp)*(a0 - ap)*dt2

       ! snap (a2) and crackle (a3) estimates for timestep selection
       a2 = ( -6.0_dp*(a0 - ap)/dt - 4.0_dp*j0 - 2.0_dp*jp ) / dt
       a3 = ( 12.0_dp*(a0 - ap)/dt + 6.0_dp*(j0 + jp) ) / dt2

       an  = norm3(ap)
       jn  = norm3(jp)
       a2n = norm3(a2)
       a3n = norm3(a3)

       ! Aarseth timestep criterion (Hermite 4th order)
       dt_new = eta * sqrt( (an*a2n + jn*jn) / max(1.0e-60_dp, (jn*a3n + a2n*a2n)) )

       new_step = quantise_block(dt_new, dt_min, max_pow)

       r(:,i) = rnew
       v(:,i) = vnew
       a(:,i) = ap
       j(:,i) = jp

       it_last(i) = it
       dt_step(i) = new_step
       it_next(i) = it + dt_step(i)

    end do
    !$omp end parallel do
  end subroutine advance_active

  subroutine acc_jerk_one(N, i, m, rpred, vpred, ai, ji)
    integer, intent(in) :: N, i
    real(dp), intent(in) :: m(N), rpred(3,N), vpred(3,N)
    real(dp), intent(out) :: ai(3), ji(3)
    integer :: k
    real(dp) :: rij(3), vij(3), r2, rinv, inv_r3, inv_r5, rv

    ai = 0.0_dp
    ji = 0.0_dp

    do k = 1, N
       if (k == i) cycle
       rij   = rpred(:,k) - rpred(:,i)
       vij   = vpred(:,k) - vpred(:,i)
       r2    = dot_product(rij, rij) + eps2
       rinv  = 1.0_dp/sqrt(r2)
       inv_r3= rinv**3
       inv_r5= inv_r3 / r2
       rv    = dot_product(rij, vij)
       ai = ai + G*m(k)*rij*inv_r3
       ji = ji + G*m(k)*( vij*inv_r3 - 3.0_dp*rv*rij*inv_r5 )
    end do
  end subroutine acc_jerk_one

  subroutine acc_jerk_all(N, m, r, v, a, j)
    integer, intent(in) :: N
    real(dp), intent(in)  :: m(N), r(3,N), v(3,N)
    real(dp), intent(out) :: a(3,N), j(3,N)
    integer :: i, k
    real(dp) :: rij(3), vij(3), r2, rinv, inv_r3, inv_r5, rv

    !$omp parallel do default(none) shared(N,m,r,v,a,j) private(i,k,rij,vij,r2,rinv,inv_r3,inv_r5,rv) schedule(static)
    do i = 1, N
       a(:,i) = 0.0_dp
       j(:,i) = 0.0_dp
       do k = 1, N
          if (k == i) cycle
          rij   = r(:,k) - r(:,i)
          vij   = v(:,k) - v(:,i)
          r2    = dot_product(rij, rij) + eps2
          rinv  = 1.0_dp/sqrt(r2)
          inv_r3= rinv**3
          inv_r5= inv_r3 / r2
          rv    = dot_product(rij, vij)
          a(:,i) = a(:,i) + G*m(k)*rij*inv_r3
          j(:,i) = j(:,i) + G*m(k)*( vij*inv_r3 - 3.0_dp*rv*rij*inv_r5 )
       end do
    end do
    !$omp end parallel do
  end subroutine acc_jerk_all

  ! -------------------- timestep logic --------------------
  subroutine init_timesteps(N, eta, dt_min, max_pow, a, j, dt_step)
    integer, intent(in) :: N, max_pow
    real(dp), intent(in) :: eta, dt_min
    real(dp), intent(in) :: a(3,N), j(3,N)
    integer(i8), intent(out) :: dt_step(N)
    integer :: i
    real(dp) :: dt_guess

    do i = 1, N
       ! Simple initial guess: eta * sqrt(|a|/|j|), with protection
       dt_guess = eta * sqrt( max(1.0e-60_dp, norm3(a(:,i))) / max(1.0e-60_dp, norm3(j(:,i))) )
       dt_step(i) = quantise_block(dt_guess, dt_min, max_pow)
    end do
  end subroutine init_timesteps

  integer(i8) function quantise_block(dt, dt_min, max_pow)
    real(dp), intent(in) :: dt, dt_min
    integer, intent(in) :: max_pow
    integer(i8) :: s, p2
    integer :: e

    s = max(1_i8, int(dt/dt_min, kind=i8))
    e = floor_log2(s)
    e = max(0, min(max_pow, e))
    p2 = ishft(1_i8, e)   ! 2^e
    quantise_block = p2
  end function quantise_block

  integer function floor_log2(x)
    integer(i8), intent(in) :: x
    integer(i8) :: y
    integer :: e

    if (x <= 1_i8) then
       floor_log2 = 0
       return
    end if

    y = x
    e = 0
    do while (y > 1_i8)
       y = ishft(y, -1)
       e = e + 1
    end do
    floor_log2 = e
  end function floor_log2

  pure real(dp) function norm3(x)
    real(dp), intent(in) :: x(3)
    norm3 = sqrt(x(1)*x(1) + x(2)*x(2) + x(3)*x(3))
  end function norm3

  ! -------------------- diagnostics & output --------------------
  subroutine kinetic_potential(N, m, r, v, T, U)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N), r(3,N), v(3,N)
    real(dp), intent(out) :: T, U
    integer :: i, k
    real(dp) :: rij(3), rnorm

    T = 0.0_dp
    do i = 1, N
       T = T + 0.5_dp*m(i)*dot_product(v(:,i), v(:,i))
    end do

    U = 0.0_dp
    do i = 1, N-1
       do k = i+1, N
          rij   = r(:,k) - r(:,i)
          rnorm = sqrt(dot_product(rij, rij) + eps2)
          U = U - G*m(i)*m(k)/rnorm
       end do
    end do
  end subroutine kinetic_potential

  real(dp) function total_energy(N, m, r, v)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N), r(3,N), v(3,N)
    real(dp) :: T, U
    call kinetic_potential(N, m, r, v, T, U)
    total_energy = T + U
  end function total_energy

  subroutine center_on_com(N, m, r, v)
    integer, intent(in)    :: N
    real(dp), intent(in)   :: m(N)
    real(dp), intent(inout):: r(3,N), v(3,N)
    integer  :: i
    real(dp) :: rcom(3), vcom(3), Mtot
    Mtot = sum(m)
    rcom = 0.0_dp
    vcom = 0.0_dp
    do i = 1, N
       rcom = rcom + m(i)*r(:,i)
       vcom = vcom + m(i)*v(:,i)
    end do
    rcom = rcom/Mtot
    vcom = vcom/Mtot
    do i = 1, N
       r(:,i) = r(:,i) - rcom
       v(:,i) = v(:,i) - vcom
    end do
  end subroutine center_on_com

  subroutine write_state_header(u, N)
    integer, intent(in) :: u, N
    integer :: k
    character(len=8) :: c
    write(u,'(A)',advance='no') 't[s]'
    do k = 1, N
       write(c,'(I0)') k
       write(u,'(A)',advance='no') ' x'//trim(c)//' y'//trim(c)//' z'//trim(c)// &
            ' vx'//trim(c)//' vy'//trim(c)//' vz'//trim(c)
    end do
    write(u,*)
  end subroutine write_state_header

  subroutine dump_state(u, N, t, r, v)
    integer, intent(in) :: u, N
    real(dp), intent(in) :: t, r(3,N), v(3,N)
    integer :: k
    write(u,'(ES25.16)', advance='no') t
    do k = 1, N
       write(u,'(6(1X,ES25.16))', advance='no') r(1,k), r(2,k), r(3,k), v(1,k), v(2,k), v(3,k)
    end do
    write(u,*)
  end subroutine dump_state

end program nbody_cluster_hermite4_block
