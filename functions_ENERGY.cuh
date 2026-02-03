////////////////////////////////////////////////////////////////////////
// calculation of enthalpy to temperature
__global__ void KERNEL_clc_TemptoEnthalpy(part1*P1,part2*P2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	//enthalpy=0.0001112*(temp*temp*temp)-0.3768*(temp*temp)+982.1*temp-828300.0;
	Real tmp_enthalpy=Ttoh(P1[i].temp,P1[i].p_type);
	P1[i].enthalpy=P2[i].enthalpy0=tmp_enthalpy;
}
////////////////////////////////////////////////////////////////////////
// calculation of enthalpy to temperature
__global__ void KERNEL_clc_EnthalpytoTemp(part1*P1)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=k_num_part2) return;
	if(P1[i].i_type==3) return;

	Real tenthalpy=P1[i].enthalpy;
	int_t tp_type=P1[i].p_type;

	P1[i].temp=htoT(tenthalpy,tp_type);
}
////////////////////////////////////////////////////////////////////////
