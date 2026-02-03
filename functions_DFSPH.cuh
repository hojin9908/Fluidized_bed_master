//#include "class Cuda_Particle_Array.h"

using namespace std;

//-------------------------------------------------------------------------------------------------
// Functions for Divergence Free SPH Calculation: Declaration
//-------------------------------------------------------------------------------------------------

void initialize_DFSPH(Cuda_Particle_Array particle_array, const Real dt, Real time);

// Pre-DFSPH functions
__global__ void KERNEL_clc_mass_sum_DFSPH(Real *rho_, Real *rho0_, uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *wij_, Real *flt_s_, int_t pnb_size, const int_t dim);
__global__ void KERNEL_clc_stiffness_DFSPH(uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *rho0_, Real *dwx_, Real *dwy_, Real *dwz_, Real *stiffness_, int_t pnb_size, const Real dt);
//__global__ void KERNEL_clc_stiffness_DFSPH(uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *x0_, Real *y0_, Real *z0_, Real *rho0_, Real *dwij_, Real *dist_, Real *stiffness_, int_t pnb_size, const Real dt);
__global__ void KERNEL_clc_predictor_DFSPH(uint_t *p_type_, Real *ux0_, Real *uy0_, Real *uz0_, Real *ux_, Real *uy_, Real *uz_, Real *ftotalx_, Real *ftotaly_, Real *ftotalz_, const Real dt, Real time);


// DFSPH calculation functions
__global__ void KERNEL_clc_pressure_force_DFSPH(Real *fpx, Real *fpy, Real *fpz, uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real* rho0_, Real *drho_, Real *stiffness_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size, const Real dt);
__global__ void KERNEL_clc_pre_velocity_DFSPH(uint_t *p_type_, Real *ux_, Real *uy_, Real *uz_, Real *fpx_, Real *fpy_, Real *fpz_, const Real dt);
__global__ void KERNEL_clc_continuity_DI(Real *drho0_, Real *drho_, uint_t *number_of_neighbors_, Real* rho0_, Real *rho_, Real *rho_err_, uint_t *pnb_, Real *m_, Real *ux_, Real *uy_, Real *uz_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size, const Real dt);
__global__ void KERNEL_clc_continuity_DF(Real *drho0_, Real *drho_, uint_t *number_of_neighbors_, Real* rho0_, Real *rho_, Real *rho_err_, uint_t *pnb_, Real *m_, Real *ux_, Real *uy_, Real *uz_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size, const Real dt);


// DFSPH presusre calculation functions
__global__ void KERNEL_clc_continuity_DI_dP(Real *drho0_, Real *drho_, Real *stiffness_, Real *p_, uint_t *number_of_neighbors_, Real *rho0_, Real *rho_, Real *rho_err_, uint_t *pnb_, Real *m_, Real *ux_, Real *uy_, Real *uz_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size, const Real dt);
__global__ void KERNEL_clc_pressure_smoothing(uint_t *p_type_, uint_t *number_of_neighbors_, Real *m_, Real *rho0_, Real* p_, uint_t *pnb_, Real *wij_, int_t pnb_size);

// Post-DFSPH functions
__global__ void KERNEL_clc_update_position_DFSPH(uint_t *p_type_, Real *x_, Real *y_, Real *z_, Real *x0_, Real *y0_, Real *z0_, Real *ux_, Real *uy_, Real *uz_, Real *ux0_, Real *uy0_, Real *uz0_, const Real dt, const Real u_limit);
__global__ void KERNEL_clc_update_position_xsph(uint_t *p_type_, uint_t *number_of_neighbors_, Real *m_, Real *flt_s_, Real *x_, Real *y_, Real *z_, Real *x0_, Real *y0_, Real *z0_, Real *ux_, Real *uy_, Real *uz_, Real *rho0_, Real* rho_, uint_t *pnb_, Real *wij_, const Real C_xsph, const Real dt, int_t pnb_size);
__global__ void KERNEL_clc_update_velocity_DFSPH(uint_t *p_type_, Real *ux_, Real *uy_, Real *uz_, Real *ux0_, Real *uy0_, Real *uz0_, Real *drho0_, Real* drho_, const Real dt, Real time, const Real u_limit);

//-------------------------------------------------------------------------------------------------
// Functions for  Divergence Free SPH Calculation: Definition
//-------------------------------------------------------------------------------------------------

// Pre-DFSPH functions
void Cuda_Particle_Array::predictor_DFSPH(const Real dt, Real time)
{
	KERNEL_clc_predictor_DFSPH << <number_of_particles, 1 >> >(p_type, ux0, uy0, uz0, ux, uy, uz, ftotalx, ftotaly, ftotalz, dt, time);
}

void Cuda_Particle_Array::calculate_stiffness_DFSPH(const Real dt)
{
	//KERNEL_clc_stiffness_DFSPH << <number_of_particles, thread_size >> >(number_of_neighbors, pnb, m, x0, y0, z0, rho0, dwij, dist, stiffness, pnb_size, dt);
	KERNEL_clc_stiffness_DFSPH << <number_of_particles, thread_size >> >(number_of_neighbors, pnb, m, rho0, dwx, dwy, dwz, stiffness, pnb_size, dt);

}

void Cuda_Particle_Array::calculate_density_DFSPH()
{
	KERNEL_clc_mass_sum_DFSPH << <number_of_particles, thread_size >> >(rho, rho0, number_of_neighbors, pnb, m, wij, flt_s, pnb_size, dim);
}

__global__ void KERNEL_clc_mass_sum_DFSPH(Real *rho_, Real *rho0_, uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *wij_, Real *flt_s_, int_t pnb_size, const int_t dim)
{
	__shared__ Real cache[1000];
	//uint_t i = blockIdx.x + blockIdx.y * gridDim.x;	// working particle index
	uint_t i = blockIdx.x;

	cache[threadIdx.x] = 0;

	//if (i < cNUM_PARTICLES[0])
	//{
	Real mj, wij;
	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;					// neighbor particle index


	number_of_neighbors = number_of_neighbors_[i];

	int_t cache_idx = threadIdx.x;

	// calculate mj * Wij
	if (cache_idx < number_of_neighbors)
	{
		j = pnb_[tid];

		mj = m_[j];
		wij = wij_[tid];

		cache[cache_idx] = mj * wij;
	}
	else
	{
		cache[cache_idx] = 0;
	}

	__syncthreads();

	// reduction (summation of mj * Wij)
	for (uint_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (cache_idx < s)
			cache[cache_idx] += cache[cache_idx + s];

		__syncthreads();
	}

	// save values to particle array
	if (cache_idx == 0)
	{
		//rho_[i] = cache[0] / flt_s_[i];
		rho0_[i] = cache[0] / flt_s_[i];
		rho_[i] = cache[0];
		//rho0_[i] = cache[0];
	}

	//}

}


//__global__ void KERNEL_clc_stiffness_DFSPH(uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *x0_, Real *y0_, Real *z0_, Real *rho0_, Real *dwij_, Real *dist_, Real *stiffness_, int_t pnb_size, const Real dt)
//{
//	__shared__ Real cache_x[1000];
//	__shared__ Real cache_y[1000];
//	__shared__ Real cache_z[1000];
//	__shared__ Real cache_dot[1000];
//
//	cache_x[threadIdx.x] = 0;
//	cache_y[threadIdx.x] = 0;
//	cache_z[threadIdx.x] = 0;
//	cache_dot[threadIdx.x] = 0;
//
//	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
//	uint_t i = blockIdx.x;
//
//	Real xi, yi, zi, xj, yj, zj;
//	Real mi, rho0;
//	Real dwij, dist;
//	Real mj;
//	Real C;
//	Real denominator, dwij_sum;
//
//	uint_t number_of_neighbors;
//	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
//	uint_t j;
//
//	int_t cache_idx = threadIdx.x;
//
//	number_of_neighbors = number_of_neighbors_[i];
//
//	if (cache_idx < number_of_neighbors)
//	{
//		j = pnb_[tid];
//
//		mj = m_[j];
//
//		xi = x0_[i];
//		yi = y0_[i];
//		zi = z0_[i];
//
//		xj = x0_[j];
//		yj = y0_[j];
//		zj = z0_[j];
//
//		dwij = dwij_[tid];
//		dist = dist_[tid];
//
//		cache_dot[cache_idx] = mj * mj * dwij * dwij;
//
//		if (dist > 0)
//		{
//			cache_x[cache_idx] = mj * dwij / dist * (xi - xj);
//			cache_y[cache_idx] = mj * dwij / dist * (yi - yj);
//			cache_z[cache_idx] = mj * dwij / dist * (zi - zj);
//		}
//		else
//		{
//			cache_x[cache_idx] = 0;
//			cache_y[cache_idx] = 0;
//			cache_z[cache_idx] = 0;
//		}
//	}
//	else
//	{
//		cache_dot[cache_idx] = 0;
//		cache_x[cache_idx] = 0;
//		cache_y[cache_idx] = 0;
//		cache_z[cache_idx] = 0;
//	}
//
//	__syncthreads();
//
//	// reduction
//	for (uint_t s = blockDim.x / 2; s > 0; s >>= 1)
//	{
//		if (cache_idx < s)
//		{
//			cache_dot[cache_idx] += cache_dot[cache_idx + s];
//			cache_x[cache_idx] += cache_x[cache_idx + s];
//			cache_y[cache_idx] += cache_y[cache_idx + s];
//			cache_z[cache_idx] += cache_z[cache_idx + s];
//		}
//
//		__syncthreads();
//	}
//
//	// calculate stiffness parameter
//	if (cache_idx == 0)
//	{
//		dwij_sum = cache_x[0] * cache_x[0] + cache_y[0] * cache_y[0] + cache_z[0] * cache_z[0];
//		denominator = fmax(dwij_sum + cache_dot[0], epsilon);
//
//		rho0 = rho0_[i];
//
//		stiffness_[i] = rho0 / denominator / dt;
//
//	}
//}
__global__ void KERNEL_clc_stiffness_DFSPH(uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real *rho0_, Real *dwx_, Real *dwy_, Real *dwz_, Real *stiffness_, int_t pnb_size, const Real dt)
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

	Real mj, rho0;
	Real dwx, dwy, dwz;
	Real denominator, dwij_sum;

	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;

	int_t cache_idx = threadIdx.x;

	number_of_neighbors = number_of_neighbors_[i];

	if (cache_idx < number_of_neighbors)
	{
		j = pnb_[tid];

		mj = m_[j];

		dwx = dwx_[tid];
		dwy = dwy_[tid];
		dwz = dwz_[tid];

		cache_dot[cache_idx] = mj * mj * (dwx * dwx + dwy * dwy + dwz * dwz);

		cache_x[cache_idx] = mj * dwx;
		cache_y[cache_idx] = mj * dwy;
		cache_z[cache_idx] = mj * dwz;
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
		denominator = fmax(dwij_sum + cache_dot[0], epsilon);

		rho0 = rho0_[i];

		stiffness_[i] = rho0 / denominator / dt;

	}
}

__global__ void KERNEL_clc_predictor_DFSPH(uint_t *p_type_, Real *ux0_, Real *uy0_, Real *uz0_, Real *ux_, Real *uy_, Real *uz_, Real *ftotalx_, Real *ftotaly_, Real *ftotalz_, const Real dt, Real time)
{
	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	//if (i < cNUM_PARTICLES[0])
	//{
	Real ux0, uy0, uz0;
	Real dux_dt0, duy_dt0, duz_dt0;

	int_t p_type_i = p_type_[i];

	switch (p_type_i)
	{
	case BOUNDARY:
		ux_[i] = 0;					// Update particle data to uncontrained x-directional velocity (initial velocity for DFSPH)			
		uy_[i] = 0;					// Update particle data to uncontrained y-directional velocity (initial velocity for DFSPH)
		uz_[i] = 0;					// Update particle data to uncontrained z-directional velocity (initial velocity for DFSPH)
		break;

	case FLUID:
		ux0 = ux0_[i];					// initial x-directional velocity
		uy0 = uy0_[i];					// initial y-directional velocity
		uz0 = uz0_[i];					// initial z-directional velocity

		dux_dt0 = ftotalx_[i];			// initial x-directional acceleration
		duy_dt0 = ftotaly_[i];			// initial y-directional acceleration
		duz_dt0 = ftotalz_[i];			// initial z-directional acceleration

		ux_[i] = ux0 + dux_dt0 * dt;					// Update particle data to uncontrained x-directional velocity (initial velocity for DFSPH)			
		uy_[i] = uy0 + duy_dt0 * dt;					// Update particle data to uncontrained y-directional velocity (initial velocity for DFSPH)
		uz_[i] = uz0 + duz_dt0 * dt;					// Update particle data to uncontrained z-directional velocity (initial velocity for DFSPH)
		break;

	case MOVING:
		ux_[i] = -0.1 * PI * sinf(PI * time); 				// Update particle data to uncontrained x-directional velocity (initial velocity for DFSPH)			
		uy_[i] = 0;												// Update particle data to uncontrained y-directional velocity (initial velocity for DFSPH)
		uz_[i] = 0;												// Update particle data to uncontrained z-directional velocity (initial velocity for DFSPH)

		break;

	default:
		break;
	}
}



// DFSPH calculation functions
void Cuda_Particle_Array::density_invariant_solver_DFSPH(const Real dt)
{
	// calculate pressure force
	KERNEL_clc_pressure_force_DFSPH << <number_of_particles, thread_size >> >(fpx, fpy, fpz, number_of_neighbors, pnb, m, rho0, drho, stiffness, dwx, dwy, dwz, pnb_size, dt);

	// predict fluid particle position
	KERNEL_clc_pre_velocity_DFSPH << <number_of_particles, 1 >> >(p_type, ux, uy, uz, fpx, fpy, fpz, dt);

	// calculate particle density and pressure
	KERNEL_clc_continuity_DI << <number_of_particles, thread_size >> >(drho0, drho, number_of_neighbors, rho0, rho, rho_err, pnb, m, ux, uy, uz, dwx, dwy, dwz, pnb_size, dt);
}

void Cuda_Particle_Array::divergence_free_solver_DFSPH(const Real dt)
{
	// calculate pressure force
	KERNEL_clc_pressure_force_DFSPH << <number_of_particles, thread_size >> >(fpx, fpy, fpz, number_of_neighbors, pnb, m, rho0, drho, stiffness, dwx, dwy, dwz, pnb_size, dt);

	// predict fluid particle position
	KERNEL_clc_pre_velocity_DFSPH << <number_of_particles, 1 >> >(p_type, ux, uy, uz, fpx, fpy, fpz, dt);

	// calculate particle density and pressure
	KERNEL_clc_continuity_DF << <number_of_particles, thread_size >> >(drho0, drho, number_of_neighbors, rho0, rho, rho_err, pnb, m, ux, uy, uz, dwx, dwy, dwz, pnb_size, dt);
}

__global__ void KERNEL_clc_pressure_force_DFSPH(Real *fpx, Real *fpy, Real *fpz, uint_t *number_of_neighbors_, uint_t *pnb_, Real *m_, Real* rho0_, Real *drho_, Real *stiffness_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size, const Real dt)
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

	Real mj, rhoi, rhoj;
	Real C_p;
	Real stiffness_i, stiffness_j;
	Real drho_i, drho_j;
	Real kappa_i, kappa_j;
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

		mj = m_[j];

		rhoi = rho0_[i];
		rhoj = rho0_[j];

		stiffness_i = stiffness_[i];
		stiffness_j = stiffness_[j];

		drho_i = drho_[i];
		drho_j = drho_[j];

		dwx = dwx_[tid];
		dwy = dwy_[tid];
		dwz = dwz_[tid];

		kappa_i = drho_i * stiffness_i;
		kappa_j = drho_j * stiffness_j;


		C_p = -mj * (kappa_i / rhoi + kappa_j / rhoj);

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

__global__ void KERNEL_clc_pre_velocity_DFSPH(uint_t *p_type_, Real *ux_, Real *uy_, Real *uz_, Real *fpx_, Real *fpy_, Real *fpz_, const Real dt)
{
	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	Real fpx, fpy, fpz;
	//Real uxp, uyp, uzp;
	Real ux, uy, uz;

	int_t p_type_i = p_type_[i];

	if (p_type_i == FLUID)
	{
		fpx = fpx_[i];
		fpy = fpy_[i];
		fpz = fpz_[i];

		ux = ux_[i];						// Predicted x-directional position by advection forces
		uy = uy_[i];						// Predicted y-directional position by advection forces
		uz = uz_[i];						// Predicted z-directional position by advection forces

		ux_[i] = ux + fpx * dt;			// Update particle position by PCISPH pressure force 
		uy_[i] = uy + fpy * dt;			// Update particle position by PCISPH pressure force 
		uz_[i] = uz + fpz * dt;			// Update particle position by PCISPH pressure force 
	}

}


//calculate continuity equation and evaluate current density--------------------------------------------------------------
__global__ void KERNEL_clc_continuity_DI(Real *drho0_, Real *drho_, uint_t *number_of_neighbors_, Real* rho0_, Real *rho_, Real *rho_err_, uint_t *pnb_, Real *m_, Real *ux_, Real *uy_, Real *uz_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size, const Real dt)
{
	__shared__ Real cache_x[1000];
	__shared__ Real cache_y[1000];
	__shared__ Real cache_z[1000];
	//__shared__ Real cache_flt[1000];

	cache_x[threadIdx.x] = 0;
	cache_y[threadIdx.x] = 0;
	cache_z[threadIdx.x] = 0;
	//cache_flt[threadIdx.x] = 0;

	//uint_t i = blockIdx.x + blockIdx.y * gridDim.x;
	uint_t i = blockIdx.x;

	//if (i < cNUM_PARTICLES[0])
	//{
	Real mi, mj, rhoi, rhoj;
	Real uxi, uxj, uyi, uyj, uzi, uzj;
	Real drho, rhopi, rho_err, rho_ref;
	//Real drho_dt;
	//Real wij;
	Real rho0_i, drho0_i, ecs;
	Real dwx, dwy, dwz;

	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;


	number_of_neighbors = number_of_neighbors_[i];

	int_t cache_idx = threadIdx.x;

	// calculate contribution of j particle on density variation (drho)
	if (cache_idx < number_of_neighbors)
	{
		// neighbor particle index
		j = pnb_[tid];

		// mass
		mj = m_[j];
		mi = m_[i];
		// density
		rhoi = rho_[i];
		rhoj = rho_[j];

		// velocity
		uxi = ux_[i];
		uyi = uy_[i];
		uzi = uz_[i];

		uxj = ux_[j];
		uyj = uy_[j];
		uzj = uz_[j];

		// kernel & distance
		dwx = dwx_[tid];
		dwy = dwy_[tid];
		dwz = dwz_[tid];

		//wij = wij_[tid];
		//cache_flt[cache_idx] = mj * wij;

		// calculate rho increment
		cache_x[cache_idx] = (uxi - uxj) * mj * dwx;			// x-directional Continuity Equation
		cache_y[cache_idx] = (uyi - uyj) * mj * dwy;			// y-directional Continuity Equation
		cache_z[cache_idx] = (uzi - uzj) * mj * dwz;			// z-directional Continuity Equation

	}
	else
	{
		cache_x[cache_idx] = 0;
		cache_y[cache_idx] = 0;
		cache_z[cache_idx] = 0;
		//cache_flt[cache_idx] = 0;
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
			//cache_flt[cache_idx] += cache_flt[cache_idx + s];
		}

		__syncthreads();
	}


	// save values to particle array
	if (cache_idx == 0)
	{
		// calculate error compensation source term
		rho_ref = 1000.;
		rho0_i = rho0_[i];
		drho0_i = drho0_[i];

		//ecs = 1 / rho_ref * (abs(rho0_i - rho_ref) * drho0_i + abs(drho0_i) * (rho0_i - rho_ref));
		//drho = cache_x[0] + cache_y[0] + cache_z[0] + ecs;
		drho = cache_x[0] + cache_y[0] + cache_z[0];

		rhopi = rhoi + drho * dt;										// predict density by Eulerian integration step		
		rho_err = rhopi - rho_ref;										// intermediate density - reference density
		//rho_err = rhoi + 0.8 * drho * dt - rho_ref;					

		rho_err = fmax(rho_err, 0);									// filterate surface particle : truncating negative pressure at surface	
		rho_err_[i] = rho_err;
		drho_[i] = rho_err / dt;									// drho_dt at current iteration

		//drho_[i] = drho_dt;
		//p_[i] += rhoi * drho_dt * stiffness_[i];					// calculate pressure equation 1 derived from PCISPH pressure force term [TRUE]
		//p_[i] += 0.5 * drho_dt * stiffness_[i] / (mi*mi);			// calculate pressure equation 2 derived from PCISPH pressure derivative term [FALSE]
		//rho_[i] = rhoi + rho_err;	
		//rho0_[i] = rhoi + rho_err;								
	}


	//}


}

__global__ void KERNEL_clc_continuity_DF(Real *drho0_, Real *drho_, uint_t *number_of_neighbors_, Real *rho0_, Real *rho_, Real *rho_err_, uint_t *pnb_, Real *m_, Real *ux_, Real *uy_, Real *uz_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size, const Real dt)
{
	__shared__ Real cache_x[1000];
	__shared__ Real cache_y[1000];
	__shared__ Real cache_z[1000];
	//__shared__ Real cache_flt[1000];

	cache_x[threadIdx.x] = 0;
	cache_y[threadIdx.x] = 0;
	cache_z[threadIdx.x] = 0;
	//cache_flt[threadIdx.x] = 0;

	uint_t i = blockIdx.x;

	//if (i < cNUM_PARTICLES[0])
	//{
	Real mj, rhoi, rhoj;
	Real uxi, uxj, uyi, uyj, uzi, uzj;
	Real drho, rhopi, rho_err, rho_ref;
	Real dwx, dwy, dwz;

	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;

	//Real wij;
	Real rho0_i, drho0_i, ecs;

	number_of_neighbors = number_of_neighbors_[i];

	int_t cache_idx = threadIdx.x;

	// calculate contribution of j particle on density variation (drho)
	if (cache_idx < number_of_neighbors)
	{
		// neighbor particle index
		j = pnb_[tid];

		// mass
		mj = m_[j];

		// density
		rhoi = rho_[i];
		rhoj = rho_[j];

		// velocity
		uxi = ux_[i];
		uyi = uy_[i];
		uzi = uz_[i];

		uxj = ux_[j];
		uyj = uy_[j];
		uzj = uz_[j];

		// kernel & distance
		dwx = dwx_[tid];
		dwy = dwy_[tid];
		dwz = dwz_[tid];

		//wij = wij_[tid];
		//cache_flt[cache_idx] = mj * wij;

		// calculate drho_dt increment
		cache_x[cache_idx] = (uxi - uxj) * mj * dwx;			// x-directional Continuity Equation
		cache_y[cache_idx] = (uyi - uyj) * mj * dwy;			// y-directional Continuity Equation
		cache_z[cache_idx] = (uzi - uzj) * mj * dwz;			// z-directional Continuity Equation

	}
	else
	{
		cache_x[cache_idx] = 0;
		cache_y[cache_idx] = 0;
		cache_z[cache_idx] = 0;
		//cache_flt[cache_idx] = 0;
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
			//cache_flt[cache_idx] += cache_flt[cache_idx + s];
		}

		__syncthreads();
	}


	// save values to particle array
	if (cache_idx == 0)
	{
		// calculate error compensation source term
		//rho0_i = rho0_[i];
		rho_ref = 1000.;
		rho0_i = rho0_[i];
		drho0_i = drho0_[i];

		//ecs = 1 / rho_ref * (abs(rho0_i - rho_ref) * drho0_i + abs(drho0_i) * (rho0_i - rho_ref));
		//drho = cache_x[0] + cache_y[0] + cache_z[0] + ecs;
		drho = cache_x[0] + cache_y[0] + cache_z[0];					// predicted density time derivative

		rhopi = rhoi + drho * dt;										// predict density by Eulerian integration step
		rho_err = rhopi - rho_ref;										// intermediate density - reference density						

		//if (rho_err < 0)	
		if (rhopi < rho_ref * 1.0)										// filterate surface particle : truncating negative pressure at surface	
		{
			rho_err = 0;
			drho = 0;
		}

		rho_err_[i] = rho_err;
		drho_[i] = drho;												// drho_dt at current iteration

		//drho_[i] = rho_err / dt;	
		//p_[i] += rhoi * rho_err / dt * stiffness_[i];					// calculate pressure equation 1 derived from PCISPH pressure force term
		//p_[i] += 0.5 * drho_dt * stiffness_[i] / (mi*mi);				// calculate pressure equation 2 derived from PCISPH pressure derivative term
		//rho_[i] = rho0_[i] + rho_err * dt;							
	}


	//}


}


// calculate pressure for DFSPH
void Cuda_Particle_Array::calculate_pressure_DFSPH(const Real dt)
{
	// calculate pressure force
	KERNEL_clc_pressure_force_DFSPH << <number_of_particles, thread_size >> >(fpx, fpy, fpz, number_of_neighbors, pnb, m, rho0, drho, stiffness, dwx, dwy, dwz, pnb_size, dt);

	// predict fluid particle position
	KERNEL_clc_pre_velocity_DFSPH << <number_of_particles, 1 >> >(p_type, ux, uy, uz, fpx, fpy, fpz, dt);

	// calculate particle density and pressure
	KERNEL_clc_continuity_DI_dP << <number_of_particles, thread_size >> >(drho0, drho, stiffness, p, number_of_neighbors, rho0, rho, rho_err, pnb, m, ux, uy, uz, dwx, dwy, dwz, pnb_size, dt);
	KERNEL_clc_pressure_smoothing << <number_of_particles, thread_size >> >(p_type, number_of_neighbors, m, rho0, p, pnb, wij, pnb_size);
}

__global__ void KERNEL_clc_continuity_DI_dP(Real *drho0_, Real *drho_, Real *stiffness_, Real *p_, uint_t *number_of_neighbors_, Real *rho0_, Real *rho_, Real *rho_err_, uint_t *pnb_, Real *m_, Real *ux_, Real *uy_, Real *uz_, Real *dwx_, Real *dwy_, Real *dwz_, int_t pnb_size, const Real dt)
{
	__shared__ Real cache_x[1000];
	__shared__ Real cache_y[1000];
	__shared__ Real cache_z[1000];

	cache_x[threadIdx.x] = 0;
	cache_y[threadIdx.x] = 0;
	cache_z[threadIdx.x] = 0;

	uint_t i = blockIdx.x;

	//if (i < cNUM_PARTICLES[0])
	//{
	Real mi, mj, rhoi, rhoj;
	Real uxi, uxj, uyi, uyj, uzi, uzj;
	Real drho, rhopi, rho_err, rho_ref;
	Real drho_dt, dpi;
	Real dwx, dwy, dwz;

	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;

	number_of_neighbors = number_of_neighbors_[i];

	int_t cache_idx = threadIdx.x;

	// calculate contribution of j particle on density variation (drho)
	if (cache_idx < number_of_neighbors)
	{
		// neighbor particle index
		j = pnb_[tid];

		// mass
		mj = m_[j];
		mi = m_[i];

		// density
		rhoi = rho_[i];
		rhoj = rho_[j];

		// velocity
		uxi = ux_[i];
		uyi = uy_[i];
		uzi = uz_[i];

		uxj = ux_[j];
		uyj = uy_[j];
		uzj = uz_[j];

		// kernel & distance
		dwx = dwx_[tid];
		dwy = dwy_[tid];
		dwz = dwz_[tid];

		// calculate rho increment
		cache_x[cache_idx] = (uxi - uxj) * mj * dwx;			// x-directional Continuity Equation
		cache_y[cache_idx] = (uyi - uyj) * mj * dwy;			// y-directional Continuity Equation
		cache_z[cache_idx] = (uzi - uzj) * mj * dwz;			// z-directional Continuity Equation
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


	// save values to particle array
	if (cache_idx == 0)
	{
		//calculate error compensation source term (excluding pressure calculation,..)
		rho_ref = 1000.;
		//rho0_i = rho0_[i];
		//drho0_i = drho0_[i];
		//ecs = 1 / rho_ref * (abs(rho0_i - rho_ref) * drho0_i + abs(drho0_i) * (rho0_i - rho_ref));
		//drho = cache_x[0] + cache_y[0] + cache_z[0] + ecs;


		drho = cache_x[0] + cache_y[0] + cache_z[0];				// predicted density time derivative

		rhopi = rhoi + drho * dt;									// predict density by Eulerian integration step
		rho_err = rhopi - rho_ref;									// intermediate density - reference density
		rho_err = fmax(rho_err, 0);									// filterate surface particle : truncating negative pressure at surface	
		rho_err_[i] = rho_err;										// Use 'rho_err' for pressure convergence criterion

		//calculate DFSPH pressure
		drho_dt = rho_err / dt;
		drho_[i] = drho_dt;
		dpi = rho_ref * drho_dt * stiffness_[i];					// calculate pressure equation(1) derived from PCISPH pressure force term [TRUE]
		p_[i] += dpi;												// pressure update

		//p_[i] += 0.5 * rhoi * drho_dt * stiffness_[i] / (mi*mi);	// calculate pressure equation(2) derived from PCISPH pressure derivative term [FALSE]
		//rho_[i] = rhopi;											
	}


	//}


}

// Post-DFSPH functions
void Cuda_Particle_Array::update_position_DFSPH(const Real dt)
{
	if (xsph_solve == 0)
	{
		KERNEL_clc_update_position_DFSPH << <number_of_particles, 1 >> >(p_type, x, y, z, x0, y0, z0, ux, uy, uz, ux0, uy0, uz0, dt, u_limit);
	}
	else
	{
		KERNEL_clc_update_position_xsph << <number_of_particles, thread_size >> >(p_type, number_of_neighbors, m, flt_s, x, y, z, x0, y0, z0, ux, uy, uz, rho0, rho, pnb, wij, C_xsph, dt, pnb_size);
	}
}

void Cuda_Particle_Array::update_velocity_DFSPH(const Real dt, Real time)
{
	KERNEL_clc_update_velocity_DFSPH << <number_of_particles, 1 >> >(p_type, ux, uy, uz, ux0, uy0, uz0, drho0, drho, dt, time, u_limit);
}

__global__ void KERNEL_clc_update_position_DFSPH(uint_t *p_type_, Real *x_, Real *y_, Real *z_, Real *x0_, Real *y0_, Real *z0_, Real *ux_, Real *uy_, Real *uz_, Real *ux0_, Real *uy0_, Real *uz0_, const Real dt, const Real u_limit)
{
	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	Real x, y, z;
	//Real x0, y0, z0;
	Real ux, uy, uz;
	//Real ux0, uy0, uz0;

	int_t p_type_i = p_type_[i];

	if (p_type_i > BOUNDARY)
	{
		x = x0_[i];
		y = y0_[i];
		z = z0_[i];

		ux = ux_[i];
		uy = uy_[i];
		uz = uz_[i];

		x += ux * dt;
		y += uy * dt;
		z += uz * dt;

		x0_[i] = x;
		y0_[i] = y;
		z0_[i] = z;

		x_[i] = x;
		y_[i] = y;
		z_[i] = z;
	}

}

__global__ void KERNEL_clc_update_position_xsph(uint_t *p_type_, uint_t *number_of_neighbors_, Real *m_, Real *flt_s_, Real *x_, Real *y_, Real *z_, Real *x0_, Real *y0_, Real *z0_, Real *ux_, Real *uy_, Real *uz_, Real *rho0_, Real* rho_, uint_t *pnb_, Real *wij_, const Real C_xsph, const Real dt, int_t pnb_size)
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

		uxi = ux_[i];
		uyi = uy_[i];
		uzi = uz_[i];

		uxj = ux_[j];
		uyj = uy_[j];
		uzj = uz_[j];

		wij = wij_[tid];

		cache_x[cache_idx] = C_xsph * mj / rhoj * (uxi - uxj) * wij / flt_si;
		cache_y[cache_idx] = C_xsph * mj / rhoj * (uyi - uyj) * wij / flt_si;
		cache_z[cache_idx] = C_xsph * mj / rhoj * (uzi - uzj) * wij / flt_si;
		//cache_flts[cache_idx] = mj / rhoj * wij;
	}
	else
	{
		cache_x[cache_idx] = 0;
		cache_y[cache_idx] = 0;
		cache_z[cache_idx] = 0;
		//cache_flts[cache_idx] = 0;
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
			//cache_flts[cache_idx] += cache_flts[cache_idx + s];
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

		x += uxi_xsph * dt * (p_type_i == 1);
		y += uyi_xsph * dt * (p_type_i == 1);
		z += uzi_xsph * dt * (p_type_i == 1);

		x0_[i] = x;
		y0_[i] = y;
		z0_[i] = z;

		x_[i] = x;
		y_[i] = y;
		z_[i] = z;

		//ux_[i] = uxi_xsph;
		//uy_[i] = uyi_xsph;
		//uz_[i] = uzi_xsph;

		//Filtering for Surface particles
		//		if (flt_si < 0.98)
		//		{
		//			uxi_xsph = uxi - cache_x[0];
		//			uyi_xsph = uyi - cache_y[0];
		//			uzi_xsph = uzi - cache_z[0];
		//
		//			x += uxi_xsph * dt * (p_type_i == 1);
		//			y += uyi_xsph * dt * (p_type_i == 1);
		//			z += uzi_xsph * dt * (p_type_i == 1);
		//		}
		//		else
		//		{
		//			x += uxi * dt * (p_type_i == 1);
		//			y += uyi * dt * (p_type_i == 1);
		//			z += uzi * dt * (p_type_i == 1);
		//		}
		//		x0_[i] = x;
		//		y0_[i] = y;
		//		z0_[i] = z;
		//
		//		x_[i] = x;
		//		y_[i] = y;
		//		z_[i] = z;		
	}
}

__global__ void KERNEL_clc_update_velocity_DFSPH(uint_t *p_type_, Real *ux_, Real *uy_, Real *uz_, Real *ux0_, Real *uy0_, Real *uz0_, Real *drho0_, Real* drho_, const Real dt, Real time, const Real u_limit)
{
	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	Real ux, uy, uz;
	int_t p_type_i = p_type_[i];

	switch (p_type_i)
	{
	case BOUNDARY:
		ux_[i] = 0;
		uy_[i] = 0;
		uz_[i] = 0;

		ux0_[i] = 0;
		uy0_[i] = 0;
		uz0_[i] = 0;
		break;

	case FLUID:
		ux = ux_[i] * p_type_i;
		uy = uy_[i] * p_type_i;
		uz = uz_[i] * p_type_i;

		if (ux*ux + uy*uy + uz*uz >u_limit * u_limit)
		{
			ux = ux0_[i];
			uy = uy0_[i];
			uz = uz0_[i];

			ux_[i] = ux;
			uy_[i] = uy;
			uz_[i] = uz;
		}
		else
		{
			ux0_[i] = ux;
			uy0_[i] = uy;
			uz0_[i] = uz;
		}
		break;

	case MOVING:
		ux_[i] = -0.1 * PI * sinf(PI * time);
		uy_[i] = 0;
		uz_[i] = 0;

		ux0_[i] = -0.1 * PI * sinf(PI * time);
		uy0_[i] = 0;
		uz0_[i] = 0;
		break;

	default:
		break;
	}

	/*if (p_type_i == BOUNDARY)
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
		ux = ux_[i];
		uy = uy_[i];
		uz = uz_[i];

		if (ux*ux + uy*uy + uz*uz >u_limit * u_limit)
		{
			ux = ux0_[i];
			uy = uy0_[i];
			uz = uz0_[i];

			ux_[i] = ux;
			uy_[i] = uy;
			uz_[i] = uz;
		}
		else
		{
			ux0_[i] = ux;
			uy0_[i] = uy;
			uz0_[i] = uz;
		}
	}
	else if (p_type_i == MOVING)
	{
		ux_[i] = -0.1 * PI * sinf(PI * time);
		uy_[i] = 0;
		uz_[i] = 0;

		ux0_[i] = -0.1 * PI * sinf(PI * time);
		uy0_[i] = 0;
		uz0_[i] = 0;
	}*/

	drho0_[i] = drho_[i];

}


void initialize_DFSPH(Cuda_Particle_Array particle_array, const Real dt, Real time)
{
	Real temp = particle_array.u_limit;
	particle_array.u_limit = 0.015;

	for (int init = 0; init < 2000; init++)
	{
		//particle_array.calculate_advforce(dt);

		particle_array.predictor_DFSPH(dt, time);

		// Density - Invariant Loop
		for (int k = 0; k < 5; k++)
		{
			particle_array.density_invariant_solver_DFSPH(dt);
		}

		particle_array.update_position_DFSPH(dt);

		particle_array.calculate_dist();

		particle_array.calculate_kernel();

		particle_array.calculate_density_DFSPH();

		//particle_array.calculate_stiffness_DFSPH(dt);

		// Divergence- Free Loop

		for (int k = 0; k < 3; k++)
		{
			particle_array.divergence_free_solver_DFSPH(dt);
		}
		particle_array.update_velocity_DFSPH(dt, time);
	}
	particle_array.u_limit = temp;
}

__global__ void KERNEL_clc_pressure_smoothing(uint_t *p_type_, uint_t *number_of_neighbors_, Real *m_, Real *rho0_, Real* p_, uint_t *pnb_, Real *wij_, int_t pnb_size)
{
	__shared__ Real cache_p[1000];
	__shared__ Real cache_flts[1000];

	cache_p[threadIdx.x] = 0;
	cache_flts[threadIdx.x] = 0;

	uint_t i = blockIdx.x;
	Real wij, mj, rhoj;
	Real pj;
	int_t p_type_i, p_type_j;

	uint_t number_of_neighbors;
	uint_t tid = threadIdx.x + blockIdx.x * pnb_size;
	uint_t j;

	int_t cache_idx = threadIdx.x;

	number_of_neighbors = number_of_neighbors_[i];

	if (cache_idx < number_of_neighbors)
	{
		j = pnb_[tid];

		mj = m_[j];
		rhoj = rho0_[j];
		pj = p_[j];

		p_type_i = p_type_[i];
		p_type_j = p_type_[j];
		wij = wij_[tid];

		//calculate pressure filtering for only neighboring fluid particles [Xiaoyang Xu, 2016]
		//calculate Shepard filter everytime for only neighboring fluid particles
		if (p_type_j == FLUID)
		{
			cache_p[cache_idx] = mj / rhoj * pj * wij;
			cache_flts[cache_idx] = mj / rhoj * wij;
		}
		else
		{
			cache_p[cache_idx] = 0;
			cache_flts[cache_idx] = 0;
		}
	}
	else
	{
		cache_p[cache_idx] = 0;
		cache_flts[cache_idx] = 0;
	}

	__syncthreads();

	// reduction
	for (uint_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (cache_idx < s)
		{
			cache_p[cache_idx] += cache_p[cache_idx + s];
			cache_flts[cache_idx] += cache_flts[cache_idx + s];
		}
		__syncthreads();
	}

	// save values
	if (cache_idx == 0)
	{
		if (p_type_i == FLUID)
		{
			p_[i] = cache_p[0] / cache_flts[0];
		}
	}
}
