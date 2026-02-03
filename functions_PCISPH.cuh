using namespace std;

//-------------------------------------------------------------------------------------------------
// Functions for Predictive-Corrective Incompressible SPH Calculation: Declaration
//-------------------------------------------------------------------------------------------------

void initialize_PCISPH(Cuda_Particle_Array particle_array, Real max_stiffness, const Real dt);

// Pre-PCISPH functions
__global__ void KERNEL_initialize_pressure(uint_t *p_type_, Real *p_, Real *fpx_, Real *fpy_, Real *fpz_, Real *ftotalx_, Real *ftotaly_, Real *ftotalz_);
__global__ void KERNEL_clc_predictor_PCISPH(uint_t *p_type_, Real *x0_, Real *y0_, Real *z0_, Real *x_adv_, Real *y_adv_, Real *z_adv_, Real *ux0_, Real *uy0_, Real *uz0_, Real *ftotalx_, Real *ftotaly_, Real *ftotalz_, const Real dt, const int_t dim);
__global__ void KERNEL_clc_stiffness_PCISPH(uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *rho0_, Real *dwx_, Real *dwy_, Real *dwz_, Real *stiffness_, int_t pnb_size, const Real dt);
//__global__ void KERNEL_clc_stiffness_PCISPH(uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *x0_, Real *y0_, Real *z0_, Real *rho0_, Real *dwij_, Real *dist_, Real *stiffness_, int_t pnb_size, const Real dt);


// PCISPH calculation functions
__global__ void KERNEL_clc_pressure_force_PCISPH(Real *fpx, Real *fpy, Real *fpz, uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *rho_, Real *p_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size);
//__global__ void KERNEL_clc_pressure_force_PCISPH(Real *fpx, Real *fpy, Real *fpz, uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *rho_, Real *x0_, Real *y0_, Real *z0_, Real *p_, Real *dwij_, Real *dist_, int_t pnb_size, const int_t dim);
__global__ void KERNEL_clc_pre_position_PCISPH(uint_t *p_type_, Real *x_, Real *y_, Real *z_, Real *x_adv_, Real *y_adv_, Real *z_adv_, Real *fpx_, Real *fpy_, Real *fpz_, const Real dt);
__global__ void KERNEL_clc_pressure_PCISPH(Real *drho_, uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *flt_s_, Real *x_, Real *y_, Real *z_, Real *rho0_, Real *rho_, Real *rho_err_, Real *p_, Real *h_, Real max_stiffness, int_t pnb_size, const Real dt, const int_t dim);


// Post-PCISPH  functions
__global__ void KERNEL_clc_update_property_PCISPH(uint_t *p_type_, Real *x_, Real *y_, Real *z_, Real *x0_, Real *y0_, Real *z0_, Real *ux_, Real *uy_, Real *uz_, Real *ux0_, Real *uy0_, Real *uz0_, Real *ftotalx_, Real *ftotaly_, Real *ftotalz_,
	Real *rho_, Real *rho0_, const Real dt, const Real u_limit);
__global__ void KERNEL_clc_update_property_xsph(uint_t *p_type_, uint_t *number_of_neighbors_, Real *m_, Real *flt_s_, Real *x_, Real *y_, Real *z_, Real *x0_, Real *y0_, Real *z0_, Real *ux_, Real *uy_, Real *uz_, Real *ux0_, Real *uy0_, Real *uz0_,
	Real *ftotalx_, Real *ftotaly_, Real *ftotalz_, Real *rho_, Real *rho0_, uint_t *pnb_, Real *wij_, const Real C_xsph, Real dt, int_t pnb_size, const Real u_limit);


//-------------------------------------------------------------------------------------------------
// Functions for Predictive-Corrective Incompressible SPH Calculation: Definition
//-------------------------------------------------------------------------------------------------

//Pre-PCISPH functions 

void Cuda_Particle_Array::calculate_advforce(const Real dt)
{
	if (fp_solve == 1)
	{
		// initialize pressure & pressure force (PCISPH preparation) 
		KERNEL_initialize_pressure << <number_of_particles, 1 >> >(p_type, p, fpx, fpy, fpz, ftotalx, ftotaly, ftotalz);
	}

	if (fv_solve == 1)
	{
		// viscous force calculation function
		//KERNEL_clc_viscous_force << <number_of_particles, thread_size >> >(fvx, fvy, fvz, number_of_neighbors, pnb, m, rho,x, y, z, ux, uy, uz, vis, dwx, dwy, dwz, dist, pnb_size, dim);
		KERNEL_add_viscous_force << <number_of_particles, thread_size >> >(ftotalx, ftotaly, ftotalz, number_of_neighbors, pnb, m, rho,
			x, y, z, ux, uy, uz, temp, dwx, dwy, dwz, dist, pnb_size, dim, p_type);
	}

	if (fva_solve == 1)
	{
		// artificial viscous force calculation function
		//KERNEL_clc_artificial_viscous_force << <number_of_particles, thread_size >> >(fvax, fvay, fvaz, number_of_neighbors, pnb, m, h, c, rho,x, y, z, ux, uy, uz, vis, dwx, dwy, dwz, dist, pnb_size, dim);
		KERNEL_add_artificial_viscous_force << <number_of_particles, thread_size >> >(ftotalx, ftotaly, ftotalz, number_of_neighbors, pnb, m, h, rho,
			x, y, z, ux, uy, uz, temp, dwx, dwy, dwz, dist, pnb_size, dim, p_type);
	}

	if (fg_solve == 1)
	{
		// gravitational force calculation function
		//KERNEL_clc_gravity_force << <number_of_particles, 1 >> >(fgx, fgy, fgz, dim);
		KERNEL_add_gravity_force << <number_of_particles, 1 >> >(ftotalx, ftotaly, ftotalz, dim);

	}

	if (fs_solve == 1)
	{
		// surface tension force calculation function
		//KERNEL_clc_surface_tension(particle_array, solv, number_of_particles, thread_size, pnb_size, dim);

	}


}

void Cuda_Particle_Array::predictor_PCISPH(const Real dt)
{
	KERNEL_clc_predictor_PCISPH << <number_of_particles, 1 >> >(p_type, x0, y0, z0, x_adv, y_adv, z_adv, ux0, uy0, uz0, ftotalx, ftotaly, ftotalz, dt, dim);
}

void Cuda_Particle_Array::calculate_stiffness(const Real dt)
{
	KERNEL_clc_stiffness_PCISPH << <number_of_particles, thread_size >> >(number_of_neighbors, pnb, m, rho0, dwx, dwy, dwz, stiffness, pnb_size, dt);
}

__global__ void KERNEL_initialize_pressure(uint_t *p_type_, Real *p_, Real *fpx_, Real *fpy_, Real *fpz_, Real *ftotalx_, Real *ftotaly_, Real *ftotalz_)
{
	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;
	int_t p_type_i = p_type_[i];

	p_[i] = 0;			// Fluid = 0 / Boundary = 100 [pa]
	fpx_[i] = 0;
	fpy_[i] = 0;
	fpz_[i] = 0;

	ftotalx_[i] = 0;
	ftotaly_[i] = 0;
	ftotalz_[i] = 0;

}

__global__ void KERNEL_clc_predictor_PCISPH(uint_t *p_type_, Real *x0_, Real *y0_, Real *z0_, Real *x_adv_, Real *y_adv_, Real *z_adv_, Real *ux0_, Real *uy0_, Real *uz0_, Real *ftotalx_, Real *ftotaly_, Real *ftotalz_, const Real dt, const int_t dim)
{
	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	//if (i < cNUM_PARTICLES[0])
	//{
	Real x0, y0, z0;
	Real ux0, uy0, uz0;
	Real dux_dt0, duy_dt0, duz_dt0;


	int_t p_type_i = p_type_[i];

	switch (p_type_i)
	{
	case BOUNDARY:
		x_adv_[i] = x0_[i];
		y_adv_[i] = y0_[i];
		z_adv_[i] = z0_[i];
		break;
	case FLUID:
		x0 = x0_[i];					// initial x-directional position
		y0 = y0_[i];					// initial y-directional position
		z0 = z0_[i];					// initial z-directional position

		ux0 = ux0_[i];					// initial x-directional velocity
		uy0 = uy0_[i];					// initial y-directional velocity
		uz0 = uz0_[i];					// initial z-directional velocity

		dux_dt0 = ftotalx_[i];			// initial x-directional acceleration
		duy_dt0 = ftotaly_[i];			// initial y-directional acceleration
		duz_dt0 = ftotalz_[i];			// initial z-directional acceleration

		x_adv_[i] = x0 + ux0 * dt + dux_dt0 * dt * dt;					// Update particle data by uncontrained x-directional position (initial position for PCISPH)			
		y_adv_[i] = y0 + uy0 * dt + duy_dt0 * dt * dt;					// Update particle data by uncontrained y-directional position (initial position for PCISPH)
		z_adv_[i] = z0 + uz0 * dt + duz_dt0 * dt * dt;					// Update particle data by uncontrained z-directional position (initial position for PCISPH)
		break;
		//case MOVING:
		//	break;
	default:
		break;
	}


}

__global__ void KERNEL_clc_stiffness_PCISPH(uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *rho0_, Real *dwx_, Real *dwy_, Real *dwz_, Real *stiffness_, int_t pnb_size, const Real dt)
{
	__shared__ Real cache_x[1000];
	__shared__ Real cache_y[1000];
	__shared__ Real cache_z[1000];
	__shared__ Real cache_dot[1000];

	cache_x[threadIdx.x] = 0;
	cache_y[threadIdx.x] = 0;
	cache_z[threadIdx.x] = 0;
	cache_dot[threadIdx.x] = 0;

	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	Real mi, rho0;
	Real dwij, dist;
	Real C;
	Real denominator, dwij_sum;
	Real dwx, dwy, dwz;

	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;

	int_t cache_idx = threadIdx.x;

	number_of_neighbors = number_of_neighbors_[i];

	if (cache_idx < number_of_neighbors)
	{
		j = pnb_[tid];

		dwx = dwx_[tid];
		dwy = dwy_[tid];
		dwz = dwz_[tid];

		cache_dot[cache_idx] = (dwx * dwx + dwy * dwy + dwz * dwz);

		cache_x[cache_idx] = dwx;
		cache_y[cache_idx] = dwy;
		cache_z[cache_idx] = dwz;
	}
	else
	{
		cache_dot[cache_idx] = 0;
		cache_x[cache_idx] = 0;
		cache_y[cache_idx] = 0;
		cache_z[cache_idx] = 0;
	}

	__syncthreads();

	// reduction
	for (uint_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (cache_idx < s)
		{
			cache_dot[cache_idx] += cache_dot[cache_idx + s];
			cache_x[cache_idx] += cache_x[cache_idx + s];
			cache_y[cache_idx] += cache_y[cache_idx + s];
			cache_z[cache_idx] += cache_z[cache_idx + s];
		}

		__syncthreads();
	}

	// calculate stiffness parameter
	if (cache_idx == 0)
	{
		dwij_sum = cache_x[0] * cache_x[0] + cache_y[0] * cache_y[0] + cache_z[0] * cache_z[0];
		denominator = dwij_sum + cache_dot[0];

		rho0 = rho0_[i];
		mi = m_[i];
		C = rho0 / dt / mi;

		stiffness_[i] = 0.5 * C * C / denominator;

	}
}




//PCISPH calculation functions

void Cuda_Particle_Array::calculate_PCISPH(const Real dt, Real max_stiffness)
{
	// calculate pressure force
	KERNEL_clc_pressure_force_PCISPH << <number_of_particles, thread_size >> >(fpx, fpy, fpz, number_of_neighbors, pnb, m, rho, p, dwx, dwy, dwz, pnb_size);

	// predict fluid particle position
	KERNEL_clc_pre_position_PCISPH << <number_of_particles, 1 >> >(p_type, x, y, z, x_adv, y_adv, z_adv, fpx, fpy, fpz, dt);

	// calculate particle density and pressure
	KERNEL_clc_pressure_PCISPH << <number_of_particles, thread_size >> >(drho, number_of_neighbors, pnb, m, flt_s, x, y, z, rho0, rho, rho_err, p, h, max_stiffness, pnb_size, dt, dim);
}

__global__ void KERNEL_clc_pressure_force_PCISPH(Real *fpx, Real *fpy, Real *fpz, uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *rho_, Real *p_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size)
{
	__shared__ Real cache_x[1000];
	__shared__ Real cache_y[1000];
	__shared__ Real cache_z[1000];

	cache_x[threadIdx.x] = 0;
	cache_y[threadIdx.x] = 0;
	cache_z[threadIdx.x] = 0;

	//uint_t i = blockIdx.x + blockIdx.y * gridDim.x;
	uint_t i = blockIdx.x;

	//if (i < cNUM_PARTICLES[0])
	//{

	Real mi, rhoi, rhoj;
	Real pi, pj;
	Real C_p;
	Real dwx, dwy, dwz;

	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;

	number_of_neighbors = number_of_neighbors_[i];

	int_t cache_idx = threadIdx.x;

	// calculate pressure force element from particle i and j
	if (cache_idx < number_of_neighbors)
	{
		j = pnb_[tid];

		mi = m_[i];

		rhoi = rho_[i];
		rhoj = rho_[j];

		pi = p_[i];
		pj = p_[j];

		dwx = dwx_[tid];
		dwy = dwy_[tid];
		dwz = dwz_[tid];

		C_p = -mi * (pi + pj) / (rhoi * rhoj);

		cache_x[cache_idx] = C_p * dwx;
		cache_y[cache_idx] = C_p * dwy;
		cache_z[cache_idx] = C_p * dwz;
	}
	else
	{
		cache_x[cache_idx] = 0;
		cache_y[cache_idx] = 0;
		cache_z[cache_idx] = 0;
	}

	__syncthreads();


	// reduction
	for (uint_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (cache_idx < s)
		{
			cache_x[cache_idx] += cache_x[cache_idx + s];
			cache_y[cache_idx] += cache_y[cache_idx + s];
			cache_z[cache_idx] += cache_z[cache_idx + s];
		}

		__syncthreads();
	}

	// save values
	if (cache_idx == 0)
	{
		fpx[i] = cache_x[0];
		fpy[i] = cache_y[0];
		fpz[i] = cache_z[0];
	}
	//}
}

__global__ void KERNEL_clc_pre_position_PCISPH(uint_t *p_type_, Real *x_, Real *y_, Real *z_, Real *x_adv_, Real *y_adv_, Real *z_adv_, Real *fpx_, Real *fpy_, Real *fpz_, const Real dt)
{
	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	Real fpx, fpy, fpz;
	Real x_adv, y_adv, z_adv;

	int_t p_type_i = p_type_[i];

	if (p_type_i == FLUID)
	{
		fpx = fpx_[i];
		fpy = fpy_[i];
		fpz = fpz_[i];

		x_adv = x_adv_[i];						// Predicted x-directional position by advection forces
		y_adv = y_adv_[i];						// Predicted y-directional position by advection forces
		z_adv = z_adv_[i];						// Predicted z-directional position by advection forces

		x_[i] = x_adv + fpx * dt * dt;			// Update particle position by PCISPH pressure force 
		y_[i] = y_adv + fpy * dt * dt;			// Update particle position by PCISPH pressure force 
		z_[i] = z_adv + fpz * dt * dt;			// Update particle position by PCISPH pressure force 
	}
}

__global__ void KERNEL_clc_pressure_PCISPH(Real *drho_, uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *flt_s_, Real *x_, Real *y_, Real *z_, Real *rho0_, Real *rho_, Real *rho_err_, Real *p_, Real *h_, Real max_stiffness, int_t pnb_size, const Real dt, const int_t dim)
{
	__shared__ Real cache[1000];

	cache[threadIdx.x] = 0;

	//uint_t i = blockIdx.x + blockIdx.y * gridDim.x;
	uint_t i = blockIdx.x;

	Real mi, mj;
	Real xi, yi, zi, xj, yj, zj;
	Real h, dist, wij, R, C;
	Real rhopi, rho_err, abs_rho_err;
	Real dpi, pi;

	Real rho_i, drho_i, ecs;

	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;

	number_of_neighbors = number_of_neighbors_[i];
	int_t cache_idx = threadIdx.x;

	//Calculate density by mass summation
	if (cache_idx < number_of_neighbors)
	{
		j = pnb_[tid];

		mi = m_[i];
		mj = m_[j];

		xi = x_[i];
		yi = y_[i];
		zi = z_[i];

		xj = x_[j];
		yj = y_[j];
		zj = z_[j];

		h = h_[i];

		dist = sqrt((xi - xj)*(xi - xj) + (yi - yj)*(yi - yj) + (zi - zj)*(zi - zj));

		R = dist / h * 0.5;

		switch (dim)
		{
		case 1:
			C = 1.25 / (2 * h);																// 5. / (4*(2h))
			wij = (R < 1) * C * (1 - R)*(1 - R)*(1 - R) * (1 + 3 * R);								// equation of Wendland 2 kernel function
			break;
		case 2:
			C = 2.228169203286535 / (4 * h * h);											// 7. / (pi *(2h)^2)  
			wij = (R < 1) * C * (1 - R)*(1 - R)*(1 - R)*(1 - R)*(1 + 4 * R);							// equation of Wendland 2 kernel function
			break;
		case 3:
			C = 3.342253804929802 / (8 * h * h * h);										// 21. / (2*pi *(2h)^3)
			wij = (R < 1) * C * (1 - R)*(1 - R)*(1 - R)*(1 - R)*(1 + 4 * R);							// equation of Wendland 2 kernel function
			break;
		}
		cache[cache_idx] = mj * wij;
	}
	else
	{
		cache[cache_idx] = 0;
	}

	__syncthreads();

	// reduction
	for (uint_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (cache_idx < s)
		{
			cache[cache_idx] += cache[cache_idx + s];
		}
		__syncthreads();
	}

	// calculate pressure and density
	if (cache_idx == 0)
	{

		//calculate error compensation source term
		//rho_i = rho_[i];
		//drho_i = drho_[i];
		//ecs = 1 / rho_ref[i] * (abs(rho_i - rho_ref[i]) * drho_i + abs(drho_i) * (rho_i - rho_ref[i]));
		//rhopi = cache[0] + ecs * dt;

		rhopi = cache[0];
		//rhopi = cache[0] / flt_s_[i];

		rho_err = rhopi - 1000.;

		rho_err = fmax(rho_err, 0);

		// dpi = stiffness * rho_err * p_relaxation;
		dpi = max_stiffness	* rho_err;
		p_[i] += dpi;
		//pi = fmax(p_[i] + dpi, 0);
		//p_[i] = pi;

		rho_err_[i] = rho_err;
		rho_[i] = rhopi;

	}

}




// Post-PCISPH functions
void Cuda_Particle_Array::update_properties_PCISPH(const Real dt)
{
	if (xsph_solve == 0)
	{
		KERNEL_clc_update_property_PCISPH << <number_of_particles, 1 >> >(p_type, x, y, z, x0, y0, z0, ux, uy, uz, ux0, uy0, uz0, ftotalx, ftotaly, ftotalz, rho, rho0, dt, u_limit);
	}
	else
	{
		KERNEL_clc_update_property_xsph << <number_of_particles, thread_size >> >(p_type, number_of_neighbors, m, flt_s, x, y, z, x0, y0, z0, ux, uy, uz, ux0, uy0, uz0, ftotalx, ftotaly, ftotalz, rho, rho0, pnb, wij, C_xsph, dt, pnb_size, u_limit);
	}
}

__global__ void KERNEL_clc_update_property_PCISPH(uint_t *p_type_, Real *x_, Real *y_, Real *z_, Real *x0_, Real *y0_, Real *z0_, Real *ux_, Real *uy_, Real *uz_, Real *ux0_, Real *uy0_, Real *uz0_, Real *ftotalx_, Real *ftotaly_, Real *ftotalz_,
	Real *rho_, Real *rho0_, const Real dt, const Real u_limit)
{
	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	Real x, y, z;
	Real ux, uy, uz;
	Real dux_dt, duy_dt, duz_dt;

	int_t p_type_i = p_type_[i];

	if (p_type_i == BOUNDARY)
	{
		ux_[i] = 0;
		uy_[i] = 0;
		uz_[i] = 0;

		ux0_[i] = 0;
		uy0_[i] = 0;
		uz0_[i] = 0;
	}
	else if (p_type_i == FLUID)
	{
		ux = ux0_[i];
		uy = uy0_[i];
		uz = uz0_[i];

		dux_dt = ftotalx_[i];
		duy_dt = ftotaly_[i];
		duz_dt = ftotalz_[i];

		ux += dux_dt * dt;
		uy += duy_dt * dt;
		uz += duz_dt * dt;

		if (ux*ux + uy*uy + uz*uz < u_limit * u_limit)
		{
			ux_[i] = ux;
			uy_[i] = uy;
			uz_[i] = uz;

			ux0_[i] = ux;
			uy0_[i] = uy;
			uz0_[i] = uz;
		}

		x0_[i] = x_[i];
		y0_[i] = y_[i];
		z0_[i] = z_[i];
	}

	// update density
	rho0_[i] = rho_[i];
}

__global__ void KERNEL_clc_update_property_xsph(uint_t *p_type_, uint_t *number_of_neighbors_, Real *m_, Real *flt_s_, Real *x_, Real *y_, Real *z_, Real *x0_, Real *y0_, Real *z0_, Real *ux_, Real *uy_, Real *uz_, Real *ux0_, Real *uy0_, Real *uz0_, 
	Real *ftotalx_, Real *ftotaly_, Real *ftotalz_, Real *rho_, Real *rho0_, uint_t *pnb_, Real *wij_, const Real C_xsph, const Real dt, int_t pnb_size, const Real u_limit)
{
	__shared__ Real cache_x[1000];
	__shared__ Real cache_y[1000];
	__shared__ Real cache_z[1000];
	__shared__ Real cache_flts[1000];

	cache_x[threadIdx.x] = 0;
	cache_y[threadIdx.x] = 0;
	cache_z[threadIdx.x] = 0;
	cache_flts[threadIdx.x] = 0;

	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	Real x, y, z;
	Real uxi, uyi, uzi, uxj, uyj, uzj;
	Real uxi_xsph, uyi_xsph, uzi_xsph;
	//Real dux_dt, duy_dt, duz_dt;
	Real wij, mj, rhoj, flt_si;
	int_t p_type_j;

	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;

	int_t p_type_i = p_type_[i];
	int_t cache_idx = threadIdx.x;

	number_of_neighbors = number_of_neighbors_[i];

	if (cache_idx < number_of_neighbors)
	{
		j = pnb_[tid];

		mj = m_[j];
		rhoj = rho0_[j];
		flt_si = flt_s_[i];
		p_type_j = p_type_[j];

		uxi = ux0_[i] * p_type_i + ftotalx_[i] * dt * p_type_i;
		uyi = uy0_[i] * p_type_i + ftotaly_[i] * dt * p_type_i;
		uzi = uz0_[i] * p_type_i + ftotalz_[i] * dt * p_type_i;

		uxj = ux0_[j] * p_type_j + ftotalx_[j] * dt * p_type_j;
		uyj = uy0_[j] * p_type_j + ftotaly_[j] * dt * p_type_j;
		uzj = uz0_[j] * p_type_j + ftotalz_[j] * dt * p_type_j;

		wij = wij_[tid];

		cache_x[cache_idx] = C_xsph * mj / rhoj * (uxi - uxj) * wij / flt_si;
		cache_y[cache_idx] = C_xsph * mj / rhoj * (uyi - uyj) * wij / flt_si;
		cache_z[cache_idx] = C_xsph * mj / rhoj * (uzi - uzj) * wij / flt_si;
		cache_flts[cache_idx] = mj / rhoj * wij;
	}
	else
	{
		cache_x[cache_idx] = 0;
		cache_y[cache_idx] = 0;
		cache_z[cache_idx] = 0;
		cache_flts[cache_idx] = 0;
	}

	__syncthreads();

	// reduction
	for (uint_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (cache_idx < s)
		{
			cache_x[cache_idx] += cache_x[cache_idx + s];
			cache_y[cache_idx] += cache_y[cache_idx + s];
			cache_z[cache_idx] += cache_z[cache_idx + s];
			cache_flts[cache_idx] += cache_flts[cache_idx + s];
		}
		__syncthreads();
	}

	// save values
	if (cache_idx == 0)
	{
		uxi_xsph = uxi - cache_x[0];
		uyi_xsph = uyi - cache_y[0];
		uzi_xsph = uzi - cache_z[0];

		x = x0_[i];
		y = y0_[i];
		z = z0_[i];

		x += uxi_xsph * dt * p_type_i;
		y += uyi_xsph * dt * p_type_i;
		z += uzi_xsph * dt * p_type_i;

		x0_[i] = x;
		y0_[i] = y;
		z0_[i] = z;

		x_[i] = x;
		y_[i] = y;
		z_[i] = z;

		if (uxi*uxi + uyi*uyi + uzi*uzi < u_limit*u_limit)
		{
			ux0_[i] = uxi;
			uy0_[i] = uyi;
			uz0_[i] = uzi;

			ux_[i] = uxi;
			uy_[i] = uyi;
			uz_[i] = uzi;
		}

		// update density
		rho0_[i] = rho_[i];

	}

}

void initialize_PCISPH(Cuda_Particle_Array particle_array, Real max_stiffness, const Real dt)
{
	Real temp = particle_array.u_limit;
	particle_array.u_limit = 0.002;

	for (int i = 0; i < 500; i++)
	{
		particle_array.calculate_dist();

		particle_array.calculate_kernel();

		particle_array.calculate_advforce(dt);

		particle_array.predictor_PCISPH(dt);

		for (int k = 0; k < 3; k++)
		{
			particle_array.calculate_PCISPH(dt, max_stiffness);
		}

		particle_array.sum_force();

		particle_array.update_properties_PCISPH(dt);
	}

	particle_array.u_limit = temp;
}
