! hermite11_4body.f90 — 11-body, 4th-order Hermite (jerk), plain-text output
program nbody_hermite_11
  implicit none
  ! ===== numerics & time =====
  integer, parameter :: dp   = selected_real_kind(15,300)
  real(dp), parameter :: G   = 6.67430e-11_dp        ! m^3 kg^-1 s^-2
  real(dp), parameter :: AU  = 1.495978707e11_dp     ! m
  real(dp), parameter :: day = 86400.0_dp
  real(dp), parameter :: year= 365.25_dp*day
  real(dp), parameter :: dt  = 50.0_dp              ! s
  real(dp), parameter :: t_end = 1000.0_dp*year        ! 15 years

  ! order & indices
  integer, parameter :: N = 11
  integer, parameter :: iSun=1, iMerc=2, iVen=3, iEarth=4, iMoon=5, iMars=6, &
                        iJup=7, iSat=8, iUra=9, iNep=10, iPlu=11

  ! ===== masses (kg) =====
  real(dp), parameter :: mSun = 1.98847e30_dp
  real(dp), parameter :: mMer = 3.3011e23_dp
  real(dp), parameter :: mVen = 4.8675e24_dp
  real(dp), parameter :: mEar = 5.9722e24_dp
  real(dp), parameter :: mMoo = 7.342e22_dp
  real(dp), parameter :: mMar = 6.4171e23_dp
  real(dp), parameter :: mJup = 1.89813e27_dp
  real(dp), parameter :: mSat = 5.6834e26_dp
  real(dp), parameter :: mUra = 8.6810e25_dp
  real(dp), parameter :: mNep = 1.02413e26_dp
  real(dp), parameter :: mPlu = 1.303e22_dp

  ! ===== semimajor axes (m) for simple circular starts =====
  real(dp), parameter :: aMer = 0.387098_dp*AU
  real(dp), parameter :: aVen = 0.723332_dp*AU
  real(dp), parameter :: aEar = 1.0_dp*AU
  real(dp), parameter :: aMar = 1.523679_dp*AU
  real(dp), parameter :: aJup = 5.2044_dp*AU
  real(dp), parameter :: aSat = 9.5826_dp*AU
  real(dp), parameter :: aUra = 19.2184_dp*AU
  real(dp), parameter :: aNep = 30.1104_dp*AU
  real(dp), parameter :: aPlu = 39.482_dp*AU
  real(dp), parameter :: aEM  = 384400.0e3_dp       ! Earth–Moon distance (m)

  ! ===== state =====
  real(dp) :: m(N)
  real(dp) :: r(3,N), v(3,N), a(3,N), j(3,N)
  real(dp) :: rpred(3,N), vpred(3,N), ap(3,N), jp(3,N)
  real(dp) :: t, E0, Ecur, dErel
  integer  :: step, nsteps, write_every

  ! files
  integer, parameter :: ustate=10, uener=11

  ! ---------- initial conditions ----------
  m = [ mSun, mMer, mVen, mEar, mMoo, mMar, mJup, mSat, mUra, mNep, mPlu ]
  r = 0.0_dp; v = 0.0_dp

  ! Place bodies in the xy plane with staggered phases.
  ! Signs/orientations chosen to spread them around.
  r(:,iSun) = [ 0.0_dp,   0.0_dp,   0.0_dp ]

  r(:,iMerc) = [ +aMer,   0.0_dp,   0.0_dp ]
  r(:,iVen ) = [ 0.0_dp,  +aVen,    0.0_dp ]
  ! Earth–Moon barycenter at (aEar,0,0):
  ! Put EM separation along +x; internal velocity along +y (counterclockwise).
  call init_earth_moon(aEar, aEM, r(:,iEarth), v(:,iEarth), r(:,iMoon), v(:,iMoon))

  r(:,iMars) = [   0.0_dp, -aMar,   0.0_dp ]
  r(:,iJup ) = [  -aJup,    0.0_dp, 0.0_dp ]
  r(:,iSat ) = [   0.0_dp, +aSat,   0.0_dp ]
  r(:,iUra ) = [ +aUra,    0.0_dp,  0.0_dp ]
  r(:,iNep ) = [  0.0_dp, -aNep,   0.0_dp ]
  r(:,iPlu ) = [ +aPlu,    0.0_dp,  0.0_dp ]

  ! circular tangential speeds about the Sun for all *non-EM* bodies
  v(:,iSun)  = [ 0.0_dp, 0.0_dp, 0.0_dp ]
  v(:,iMerc) = [ 0.0_dp,  sqrt(G*(mSun+mMer)/aMer),              0.0_dp ]
  v(:,iVen ) = [ -sqrt(G*(mSun+mVen)/aVen),                       0.0_dp, 0.0_dp ]
  ! Earth & Moon velocities have already been set by init_earth_moon
  v(:,iMars) = [  sqrt(G*(mSun+mMar)/aMar),                       0.0_dp, 0.0_dp ]
  v(:,iJup ) = [  0.0_dp, -sqrt(G*(mSun+mJup)/aJup),              0.0_dp ]
  v(:,iSat ) = [ -sqrt(G*(mSun+mSat)/aSat),                       0.0_dp, 0.0_dp ]
  v(:,iUra ) = [  0.0_dp,  sqrt(G*(mSun+mUra)/aUra),              0.0_dp ]
  v(:,iNep ) = [  sqrt(G*(mSun+mNep)/aNep),                       0.0_dp, 0.0_dp ]
  v(:,iPlu ) = [  0.0_dp,  sqrt(G*(mSun+mPlu)/aPlu),              0.0_dp ]

  ! shift to barycentric frame (COM at origin; zero total momentum)
  call center_on_com(N, m, r, v)

  ! initial field
  call acc_jerk(N, m, r, v, a, j)

  ! ---------- output ----------
  open(ustate, file='state.txt',  status='replace', action='write', form='formatted')
  call write_state_header(ustate, N)

  open(uener,  file='energy.txt', status='replace', action='write', form='formatted')
  write(uener,'(A)') 't[s] E[J] dE/E0'

  E0 = total_energy(N, m, r, v)

  t           = 0.0_dp
  nsteps      = int(t_end/dt)
  write_every = int(12.0_dp*3600.0_dp/dt)   ! every 12 hours

  call dump_state(ustate, N, t, r, v)
  write(uener,'(ES25.16,2(1X,ES25.16))') t, E0, 0.0_dp

  ! ---------- main loop (4th-order Hermite) ----------
  do step = 1, nsteps
     rpred = r + v*dt + 0.5_dp*a*dt*dt + (1.0_dp/6.0_dp)*j*dt*dt*dt
     vpred = v + a*dt + 0.5_dp*j*dt*dt

     call acc_jerk(N, m, rpred, vpred, ap, jp)

     v = v + 0.5_dp*(a+ap)*dt + (1.0_dp/12.0_dp)*(j - jp)*dt*dt
     r = r + 0.5_dp*(vpred + v)*dt + (1.0_dp/12.0_dp)*(a - ap)*dt*dt

     call acc_jerk(N, m, r, v, a, j)

     t = t + dt

     if (mod(step, write_every) == 0) then
        Ecur  = total_energy(N, m, r, v)
        dErel = (Ecur - E0)/E0
        call dump_state(ustate, N, t, r, v)
        write(uener,'(ES25.16,2(1X,ES25.16))') t, Ecur, dErel
     end if
  end do

  close(ustate); close(uener)

contains
  ! ---------- Earth–Moon barycentric initializer ----------
  subroutine init_earth_moon(a_bary, a_em, rE, vE, rM, vM)
    real(dp), intent(in)  :: a_bary, a_em
    real(dp), intent(out) :: rE(3), vE(3), rM(3), vM(3)
    real(dp) :: mu_sun, mu_em, fE, fM, v_b(3), d(3), vrel(3)

    fE = mMoo/(mEar + mMoo)       ! Earth offset fraction
    fM = mEar/(mEar + mMoo)       ! Moon  offset fraction

    ! EM barycenter at (+a_bary,0,0), moving +y
    mu_sun = G*(mSun + mEar + mMoo)
    v_b = [ 0.0_dp, sqrt(mu_sun/a_bary), 0.0_dp ]

    ! Internal EM circular motion: separation along +x, velocity +y
    d    = [ a_em, 0.0_dp, 0.0_dp ]
    mu_em = G*(mEar + mMoo)
    vrel  = [ 0.0_dp, sqrt(mu_em/a_em), 0.0_dp ]

    rE = [ a_bary, 0.0_dp, 0.0_dp ] - fE*d
    rM = [ a_bary, 0.0_dp, 0.0_dp ] + fM*d

    vE = v_b - fE*vrel
    vM = v_b + fM*vrel
  end subroutine init_earth_moon

  ! ---------- acceleration & jerk ----------
  subroutine acc_jerk(N, m, r, v, a, j)
    integer, intent(in) :: N
    real(dp), intent(in)  :: m(N), r(3,N), v(3,N)
    real(dp), intent(out) :: a(3,N), j(3,N)
    integer :: i, k
    real(dp) :: rij(3), vij(3), r2, rinv, inv_r3, inv_r5, rv

    a = 0.0_dp; j = 0.0_dp
    do i = 1, N
       do k = 1, N
          if (k == i) cycle
          rij  = r(:,k) - r(:,i)
          vij  = v(:,k) - v(:,i)
          r2   = dot_product(rij, rij)
          rinv = 1.0_dp/sqrt(r2)
          inv_r3 = rinv**3
          inv_r5 = inv_r3 / r2
          rv   = dot_product(rij, vij)

          a(:,i) = a(:,i) + G*m(k)*rij*inv_r3
          j(:,i) = j(:,i) + G*m(k)*( vij*inv_r3 - 3.0_dp*rv*rij*inv_r5 )
       end do
    end do
  end subroutine acc_jerk

  ! ---------- total energy ----------
  function total_energy(N, m, r, v) result(E)
    integer, intent(in) :: N
    real(dp), intent(in) :: m(N), r(3,N), v(3,N)
    real(dp) :: E, T, U
    integer  :: i, k
    real(dp) :: rij(3), rnorm

    T = 0.0_dp
    do i = 1, N
       T = T + 0.5_dp*m(i)*dot_product(v(:,i), v(:,i))
    end do

    U = 0.0_dp
    do i = 1, N-1
       do k = i+1, N
          rij   = r(:,k) - r(:,i)
          rnorm = sqrt(dot_product(rij, rij))
          U = U - G*m(i)*m(k)/rnorm
       end do
    end do
    E = T + U
  end function total_energy

  ! ---------- shift to system barycenter ----------
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

  ! ---------- header writer (space-delimited, adapts to N) ----------
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

  ! ---------- one-line, space-delimited state writer (general N) ----------
  subroutine dump_state(u, N, t, r, v)
    integer, intent(in) :: u, N
    real(dp), intent(in) :: t, r(3,N), v(3,N)
    real(dp) :: line(1 + 6*N)
    integer  :: k, idx
    line(1) = t
    idx = 2
    do k = 1, N
       line(idx:idx+5) = [ r(1,k), r(2,k), r(3,k), v(1,k), v(2,k), v(3,k) ]
       idx = idx + 6
    end do
    write(u,'(ES25.16,*(1X,ES25.16))') line   ! unlimited repeat, adapts to N
  end subroutine dump_state
end program nbody_hermite_11
