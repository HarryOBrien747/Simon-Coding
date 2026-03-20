!===============================================================================
! 1000-star cluster with 4th-order Hermite + individual timesteps + OpenMP
!===============================================================================
program cluster_hermite4_individual_dt
  use, intrinsic :: iso_fortran_env, only: output_unit
  use omp_lib
  implicit none

  ! ===== Precision & constants =====
  integer, parameter :: dp = selected_real_kind(15,300)
  real(dp), parameter :: G    = 6.67430e-11_dp              ! m^3 kg^-1 s^-2
  real(dp), parameter :: pc   = 3.0857e16_dp                ! parsec in m
  real(dp), parameter :: Msun = 1.98847e30_dp               ! solar mass in kg
  real(dp), parameter :: year = 365.25_dp * 86400.0_dp      ! year in seconds
  real(dp), parameter :: pi   = 3.14159265358979323846_dp

  ! ===== Simulation parameters =====
  integer, parameter :: N = 1000                            ! number of stars
  real(dp), parameter :: R_cluster = 1.0_dp * pc            ! cluster radius (1 pc)
  real(dp), parameter :: v_scale   = 1.0e3_dp               ! velocity scale ~1 km/s
  real(dp), parameter :: t_end     = 100.0_dp * year        ! simulation time
  real(dp), parameter :: eta       = 0.01_dp                ! timestep parameter
  real(dp), parameter :: dt_base   = 1000.0_dp              ! base timestep (s)
  
  ! ===== State arrays =====
  real(dp) :: m(N)                    ! masses
  real(dp) :: r(3,N), v(3,N)          ! positions, velocities
  real(dp) :: a(3,N), j(3,N)          ! accelerations, jerks
  real(dp) :: a0(3,N), j0(3,N)        ! saved values from last full step
  real(dp) :: dt(N)                   ! individual timesteps
  real(dp) :: t_next(N)               ! next time for each particle
  integer  :: level(N)                ! timestep level (power of 2)
  
  ! Temporary arrays for prediction
  real(dp) :: rpred(3,N), vpred(3,N), apred(3,N), jpred(3,N)
  
  ! ===== Time & I/O =====
  real(dp) :: t_global, E0, Ecur, dErel
  integer  :: step, nsteps_base, io_counter
  integer, parameter :: ustate=10, uener=11, ulog=12
  integer :: i, ios
  character(len=32) :: tag
  character(len=128) :: state_name, energy_name, log_name

  ! ===== Initialize =====
  print *, "=========================================="
  print *, "  1000-star cluster simulation"
  print *, "  4th-order Hermite + individual dt"
  print *, "=========================================="
  print *, ""
  
  ! Generate initial conditions
  call init_star_cluster(N, m, r, v, R_cluster, v_scale)
  print *, "Initial conditions generated."
  print *, "  Mass range:", minval(m)/Msun, "to", maxval(m)/Msun, "Msun"
  
  ! Virialize (apply virial theorem to scale velocities)
  call virialize(N, m, r, v)
  print *, "Cluster virialized."
  
  ! Initial accelerations and jerks
  call compute_acc_jerk_all(N, m, r, v, a, j)
  a0 = a
  j0 = j
  
  ! Assign individual timesteps
  call assign_timesteps(N, a, j, dt, level, dt_base, eta)
  print *, "Individual timesteps assigned."
  print *, "  dt range:", minval(dt), "to", maxval(dt), "s"
  print *, "  dt range:", minval(dt)/year, "to", maxval(dt)/year, "years"
  
  ! Open output files
  call timestamp_tag(tag)
  state_name  = 'cluster_state_'  // trim(tag) // '.txt'
  energy_name = 'cluster_energy_' // trim(tag) // '.txt'
  log_name    = 'cluster_log_'    // trim(tag) // '.txt'
  
  open(ustate, file=state_name,  status='replace', iostat=ios)
  if (ios /= 0) stop 'Failed to open state file'
  call write_state_header(ustate, N)
  
  open(uener, file=energy_name, status='replace', iostat=ios)
  if (ios /= 0) stop 'Failed to open energy file'
  write(uener,'(A)') '# t[yr]  E[J]  dE/E0'
  
  open(ulog, file=log_name, status='replace', iostat=ios)
  if (ios /= 0) stop 'Failed to open log file'
  
  E0 = total_energy(N, m, r, v)
  t_global = 0.0_dp
  t_next = dt  ! next update time for each particle
  
  ! Write initial state
  call dump_state(ustate, N, t_global, r, v)
  write(uener,'(3(ES25.16,1X))') t_global/year, E0, 0.0_dp
  
  print *, ""
  print *, "Initial energy:", E0, "J"
  print *, "OpenMP: using", omp_get_max_threads(), "threads"
  print *, ""
  print *, "Starting integration..."
  
  nsteps_base = int(t_end / dt_base) + 1
  io_counter = 0
  step = 0
  
  ! ===== Main integration loop =====
  do while (t_global < t_end)
    step = step + 1
    
    ! Find minimum next time
    t_global = minval(t_next)
    
    ! Update particles that need updating
    call update_particles(N, m, r, v, a, j, a0, j0, dt, t_next, t_global, level, eta)
    
    ! Periodic output every ~0.1 years
    if (mod(step, 1000) == 0 .or. t_global >= t_end) then
      io_counter = io_counter + 1
      Ecur = total_energy(N, m, r, v)
      dErel = (Ecur - E0) / E0
      
      call dump_state(ustate, N, t_global, r, v)
      write(uener,'(3(ES25.16,1X))') t_global/year, Ecur, dErel
      
      if (mod(io_counter, 10) == 0) then
        write(ulog,'(A,F10.4,A,ES12.4)') 'Time: ', t_global/year, &
              ' yr,  dE/E0 = ', dErel
        print '(A,F10.4,A,ES12.4,A,I0)', 'Time: ', t_global/year, &
              ' yr,  dE/E0 = ', dErel, ',  step: ', step
      end if
    end if
    
    if (t_global >= t_end) exit
  end do
  
  print *, ""
  print *, "Integration complete!"
  print *, "Final time:", t_global/year, "years"
  print *, "Total steps:", step
  
  close(ustate)
  close(uener)
  close(ulog)
  
contains

  !-----------------------------------------------------------------------------
  ! Generate initial star cluster with IMF
  !-----------------------------------------------------------------------------
  subroutine init_star_cluster(N, m, r, v, R, v_rms)
    integer, intent(in) :: N
    real(dp), intent(out) :: m(N), r(3,N), v(3,N)
    real(dp), intent(in) :: R, v_rms
    integer :: i
    real(dp) :: theta, phi, rad, u1, u2, u3
    
    ! Seed random number generator
    call random_seed()
    
    ! Assign masses using Kroupa IMF (0.2 to 4 Msun)
    do i = 1, N
      m(i) = sample_kroupa_imf() * Msun
    end do
    
    ! Random positions in a sphere (uniform in volume)
    do i = 1, N
      call random_number(u1)
      call random_number(u2)
      call random_number(u3)
      
      ! Uniform in sphere
      rad   = R * u1**(1.0_dp/3.0_dp)
      theta = acos(2.0_dp*u2 - 1.0_dp)
      phi   = 2.0_dp * pi * u3
      
      r(1,i) = rad * sin(theta) * cos(phi)
      r(2,i) = rad * sin(theta) * sin(phi)
      r(3,i) = rad * cos(theta)
    end do
    
    ! Random velocities (Gaussian with rms = v_rms)
    do i = 1, N
      v(1,i) = v_rms * random_gaussian()
      v(2,i) = v_rms * random_gaussian()
      v(3,i) = v_rms * random_gaussian()
    end do
    
    ! Center on COM
    call center_on_com(N, m, r, v)
  end subroutine init_star_cluster

  !-----------------------------------------------------------------------------
  ! Sample mass from Kroupa IMF (0.2 to 4 Msun)
  ! dN/dm ∝ m^(-2.3) for m > 0.5, m^(-1.3) for 0.08 < m < 0.5
  !-----------------------------------------------------------------------------
  function sample_kroupa_imf() result(mass)
    real(dp) :: mass, u, alpha1, alpha2, m_break, m_min, m_max
    real(dp) :: norm1, norm2, frac1
    
    m_min = 0.2_dp
    m_break = 0.5_dp
    m_max = 4.0_dp
    alpha1 = -1.3_dp
    alpha2 = -2.3_dp
    
    ! Normalization
    norm1 = (m_break**(alpha1+1.0_dp) - m_min**(alpha1+1.0_dp)) / (alpha1+1.0_dp)
    norm2 = (m_max**(alpha2+1.0_dp) - m_break**(alpha2+1.0_dp)) / (alpha2+1.0_dp)
    frac1 = norm1 / (norm1 + norm2)
    
    call random_number(u)
    
    if (u < frac1) then
      ! Lower mass range
      call random_number(u)
      mass = (u * (m_break**(alpha1+1.0_dp) - m_min**(alpha1+1.0_dp)) + &
              m_min**(alpha1+1.0_dp))**(1.0_dp/(alpha1+1.0_dp))
    else
      ! Higher mass range
      call random_number(u)
      mass = (u * (m_max**(alpha2+1.0_dp) - m_break**(alpha2+1.0_dp)) + &
              m_break**(alpha2+1.0_dp))**(1.0_dp/(alpha2+1.0_dp))
    end if
  end function sample_kroupa_imf

  !-----------------------------------------------------------------------------
  ! Generate Gaussian random number (Box-Muller)
  !-----------------------------------------------------------------------------
  function random_gaussian() result(x)
    real(dp) :: x, u1, u2
    call random_number(u1)
    call random_number(u2)
    x = sqrt(-2.0_dp * log(u1)) * cos(2.0_dp * pi * u2)
  end function random_gaussian

  !-----------------------------------------------------------------------------
  ! Virialize cluster: scale velocities so 2T + U = 0
  !-----------------------------------------------------------------------------
  subroutine virialize(N, m, r, v)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N), r(3,N)
    real(dp), intent(inout) :: v(3,N)
    real(dp) :: T, U, Q, scale_factor
    integer :: i, k
    real(dp) :: rij(3), rnorm
    
    ! Kinetic energy
    T = 0.0_dp
    do i = 1, N
      T = T + 0.5_dp * m(i) * dot_product(v(:,i), v(:,i))
    end do
    
    ! Potential energy
    U = 0.0_dp
    do i = 1, N-1
      do k = i+1, N
        rij = r(:,k) - r(:,i)
        rnorm = sqrt(dot_product(rij, rij))
        U = U - G * m(i) * m(k) / rnorm
      end do
    end do
    
    ! Virial ratio Q = T / |U|, want Q = 0.5
    Q = T / abs(U)
    scale_factor = sqrt(0.5_dp / Q)
    
    print *, "  Pre-virial Q =", Q
    
    ! Scale velocities
    v = v * scale_factor
    
    ! Verify
    T = 0.0_dp
    do i = 1, N
      T = T + 0.5_dp * m(i) * dot_product(v(:,i), v(:,i))
    end do
    print *, "  Post-virial Q =", T / abs(U)
  end subroutine virialize

  !-----------------------------------------------------------------------------
  ! Assign individual timesteps based on acceleration and jerk
  !-----------------------------------------------------------------------------
  subroutine assign_timesteps(N, a, j, dt, level, dt_base, eta)
    integer, intent(in) :: N
    real(dp), intent(in) :: a(3,N), j(3,N), dt_base, eta
    real(dp), intent(out) :: dt(N)
    integer, intent(out) :: level(N)
    integer :: i, lev
    real(dp) :: a_mag, j_mag, dt_ideal, dt_max
    
    dt_max = dt_base
    
    do i = 1, N
      a_mag = sqrt(dot_product(a(:,i), a(:,i)))
      j_mag = sqrt(dot_product(j(:,i), j(:,i)))
      
      if (a_mag > 0.0_dp .and. j_mag > 0.0_dp) then
        ! Timestep criterion: dt = eta * sqrt(|a| / |j|)
        dt_ideal = eta * sqrt(a_mag / j_mag)
      else
        dt_ideal = dt_max
      end if
      
      ! Round down to nearest power-of-2 fraction of dt_base
      lev = max(0, int(log(dt_max / dt_ideal) / log(2.0_dp)))
      level(i) = min(lev, 20)  ! cap at 2^20 ~ million
      dt(i) = dt_max / (2.0_dp**level(i))
    end do
  end subroutine assign_timesteps

  !-----------------------------------------------------------------------------
  ! Update particles whose time has come
  !-----------------------------------------------------------------------------
  subroutine update_particles(N, m, r, v, a, j, a0, j0, dt, t_next, &
                               t_global, level, eta)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N)
    real(dp), intent(inout) :: r(3,N), v(3,N), a(3,N), j(3,N)
    real(dp), intent(inout) :: a0(3,N), j0(3,N), dt(N), t_next(N)
    integer, intent(inout) :: level(N)
    real(dp), intent(in) :: t_global, eta
    integer :: i
    logical :: active(N)
    real(dp) :: dt_loc, vold(3)
    
    ! Identify active particles
    active = (abs(t_next - t_global) < 1.0e-10_dp)
    
    ! Predict positions/velocities for ALL particles to t_global
    !$omp parallel do default(none) shared(N,r,v,a,j,t_next,t_global,rpred,vpred) &
    !$omp private(i,dt_loc) schedule(static)
    do i = 1, N
      dt_loc = t_global - (t_next(i) - dt(i))
      rpred(:,i) = r(:,i) + v(:,i)*dt_loc + 0.5_dp*a(:,i)*dt_loc**2 + &
                   (1.0_dp/6.0_dp)*j(:,i)*dt_loc**3
      vpred(:,i) = v(:,i) + a(:,i)*dt_loc + 0.5_dp*j(:,i)*dt_loc**2
    end do
    !$omp end parallel do
    
    ! Compute accelerations for active particles at predicted positions
    call compute_acc_jerk_active(N, m, rpred, vpred, apred, jpred, active)
    
    ! Update active particles with 4th-order Hermite corrector
    !$omp parallel do default(none) &
    !$omp shared(N,active,r,v,a,j,a0,j0,apred,jpred,dt,t_next,level,eta) &
    !$omp private(i,dt_loc,vold) schedule(dynamic)
    do i = 1, N
      if (.not. active(i)) cycle
      
      dt_loc = dt(i)
      
      ! Save old velocity
      vold = v(:,i)
      
      ! 4th-order Hermite corrector
      v(:,i) = v(:,i) + 0.5_dp*(a(:,i) + apred(:,i))*dt_loc + &
               (1.0_dp/12.0_dp)*(j(:,i) - jpred(:,i))*dt_loc**2
      r(:,i) = r(:,i) + 0.5_dp*(vold + v(:,i))*dt_loc + &
               (1.0_dp/12.0_dp)*(a(:,i) - apred(:,i))*dt_loc**2
      
      ! Update acceleration and jerk
      a0(:,i) = a(:,i)
      j0(:,i) = j(:,i)
      a(:,i) = apred(:,i)
      j(:,i) = jpred(:,i)
      
      ! Update timestep based on new a, j
      call update_single_timestep(i, a, j, dt, level, eta)
      
      ! Advance next time
      t_next(i) = t_next(i) + dt(i)
    end do
    !$omp end parallel do
  end subroutine update_particles

  !-----------------------------------------------------------------------------
  ! Compute accelerations for all particles
  !-----------------------------------------------------------------------------
  subroutine compute_acc_jerk_all(N, m, r, v, a, j)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N), r(3,N), v(3,N)
    real(dp), intent(out) :: a(3,N), j(3,N)
    integer :: i, k
    real(dp) :: rij(3), vij(3), r2, rinv, inv_r3, inv_r5, rv
    
    !$omp parallel do default(none) shared(N,m,r,v,a,j) &
    !$omp private(i,k,rij,vij,r2,rinv,inv_r3,inv_r5,rv) schedule(static)
    do i = 1, N
      a(:,i) = 0.0_dp
      j(:,i) = 0.0_dp
      do k = 1, N
        if (k == i) cycle
        rij = r(:,k) - r(:,i)
        vij = v(:,k) - v(:,i)
        r2 = dot_product(rij, rij)
        rinv = 1.0_dp / sqrt(r2)
        inv_r3 = rinv**3
        inv_r5 = inv_r3 / r2
        rv = dot_product(rij, vij)
        a(:,i) = a(:,i) + G*m(k)*rij*inv_r3
        j(:,i) = j(:,i) + G*m(k)*(vij*inv_r3 - 3.0_dp*rv*rij*inv_r5)
      end do
    end do
    !$omp end parallel do
  end subroutine compute_acc_jerk_all

  !-----------------------------------------------------------------------------
  ! Compute accelerations only for active particles
  !-----------------------------------------------------------------------------
  subroutine compute_acc_jerk_active(N, m, r, v, a, j, active)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N), r(3,N), v(3,N)
    real(dp), intent(out) :: a(3,N), j(3,N)
    logical, intent(in) :: active(N)
    integer :: i, k
    real(dp) :: rij(3), vij(3), r2, rinv, inv_r3, inv_r5, rv
    
    !$omp parallel do default(none) shared(N,m,r,v,a,j,active) &
    !$omp private(i,k,rij,vij,r2,rinv,inv_r3,inv_r5,rv) schedule(dynamic)
    do i = 1, N
      if (.not. active(i)) then
        a(:,i) = 0.0_dp
        j(:,i) = 0.0_dp
        cycle
      end if
      
      a(:,i) = 0.0_dp
      j(:,i) = 0.0_dp
      do k = 1, N
        if (k == i) cycle
        rij = r(:,k) - r(:,i)
        vij = v(:,k) - v(:,i)
        r2 = dot_product(rij, rij)
        rinv = 1.0_dp / sqrt(r2)
        inv_r3 = rinv**3
        inv_r5 = inv_r3 / r2
        rv = dot_product(rij, vij)
        a(:,i) = a(:,i) + G*m(k)*rij*inv_r3
        j(:,i) = j(:,i) + G*m(k)*(vij*inv_r3 - 3.0_dp*rv*rij*inv_r5)
      end do
    end do
    !$omp end parallel do
  end subroutine compute_acc_jerk_active

  !-----------------------------------------------------------------------------
  ! Update timestep for single particle
  !-----------------------------------------------------------------------------
  subroutine update_single_timestep(i, a, j, dt, level, eta)
    integer, intent(in) :: i
    real(dp), intent(in) :: a(3,N), j(3,N), eta
    real(dp), intent(inout) :: dt(N)
    integer, intent(inout) :: level(N)
    real(dp) :: a_mag, j_mag, dt_ideal, dt_max, dt_base_loc
    integer :: lev
    
    dt_base_loc = dt_base
    dt_max = dt_base_loc
    
    a_mag = sqrt(dot_product(a(:,i), a(:,i)))
    j_mag = sqrt(dot_product(j(:,i), j(:,i)))
    
    if (a_mag > 0.0_dp .and. j_mag > 0.0_dp) then
      dt_ideal = eta * sqrt(a_mag / j_mag)
    else
      dt_ideal = dt_max
    end if
    
    lev = max(0, int(log(dt_max / dt_ideal) / log(2.0_dp)))
    level(i) = min(lev, 20)
    dt(i) = dt_max / (2.0_dp**level(i))
  end subroutine update_single_timestep

  !-----------------------------------------------------------------------------
  ! Total energy
  !-----------------------------------------------------------------------------
  function total_energy(N, m, r, v) result(E)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N), r(3,N), v(3,N)
    real(dp) :: E, T, U
    integer :: i, k
    real(dp) :: rij(3), rnorm
    
    T = 0.0_dp
    do i = 1, N
      T = T + 0.5_dp*m(i)*dot_product(v(:,i), v(:,i))
    end do
    
    U = 0.0_dp
    do i = 1, N-1
      do k = i+1, N
        rij = r(:,k) - r(:,i)
        rnorm = sqrt(dot_product(rij, rij))
        U = U - G*m(i)*m(k)/rnorm
      end do
    end do
    
    E = T + U
  end function total_energy

  !-----------------------------------------------------------------------------
  ! Center on center of mass
  !-----------------------------------------------------------------------------
  subroutine center_on_com(N, m, r, v)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N)
    real(dp), intent(inout) :: r(3,N), v(3,N)
    integer :: i
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

  !-----------------------------------------------------------------------------
  ! I/O routines
  !-----------------------------------------------------------------------------
  subroutine write_state_header(u, N)
    integer, intent(in) :: u, N
    integer :: k
    character(len=8) :: c
    write(u,'(A)',advance='no') '# t[s]'
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
    real(dp) :: line(1 + 6*N)
    integer :: k, idx
    line(1) = t
    idx = 2
    do k = 1, N
      line(idx:idx+5) = [ r(1,k), r(2,k), r(3,k), v(1,k), v(2,k), v(3,k) ]
      idx = idx + 6
    end do
    write(u,'(*(ES16.8,1X))') line
  end subroutine dump_state

  subroutine timestamp_tag(tag)
    character(len=*), intent(out) :: tag
    integer :: v(8)
    call date_and_time(values=v)
    write(tag,'(I4.4,I2.2,I2.2,"_",I2.2,I2.2,I2.2)') v(1),v(2),v(3),v(5),v(6),v(7)
  end subroutine timestamp_tag

end program cluster_hermite4_individual_dt