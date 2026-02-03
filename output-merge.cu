#include<stdio.h>
#include<stdlib.h>
#include<string.h>
//////////////////////////////////////////////////////////////////////////////
/*
int main(){
	FILE*r1,*r2;
	FILE*w;
	int i,j,pi;
	char FileName1[256],FileName2[256],FileNamew[256];
	char dummy[512];
	int rp1,rp2,wp;
	float x,y,z;

	sprintf(FileName1,"./plotdata/fluid_0stp-0.vtk");
	sprintf(FileName2,"./plotdata/fluid_0stp-1.vtk");
	sprintf(FileNamew,"./merge/fluid_0stp.vtk");

	r1=fopen(FileName1,"r+");
	r2=fopen(FileName2,"r+");
	w=fopen(FileNamew,"w+");

	fprintf(w,"# vtk DataFile Version 3.0\n");
	fprintf(w,"Print out results in vtk format\n");
	fprintf(w,"ASCII\n");
	fprintf(w,"DATASET POLYDATA\n");

	fscanf(r1,"%d\n",&rp1);
	fscanf(r2,"%d\n",&rp2);
	wp=rp1+rp2;
	fprintf(w,"POINTS\t%d\tfloat\n",wp);					// define particles position as POINTS

	for(i=0;i<rp1;i++){
		fscanf(r1,"%f\t%f\t%f\t\n",&x,&y,&z);
		fprintf(w,"%f\t%f\t%f\t\n",x,y,z);
	}
	for(i=0;i<rp2;i++){
		fscanf(r2,"%f\t%f\t%f\t\n",&x,&y,&z);
		fprintf(w,"%f\t%f\t%f\t\n",x,y,z);
	}
	fprintf(w,"VERTICES\t%d\t%d\n",wp,2*wp);
	for(i=0;i<wp;i++){
		fprintf(w,"1\t%d\n",i);
	}
	fprintf(w,"POINT_DATA\t%d\n",wp);
	fprintf(w,"FIELD FieldData\t1\n");
	fprintf(w,"p-type\t1\t%d\tfloat\n",wp);

	fscanf(r1,"%s\n",&dummy);
	fscanf(r1,"%s\n",&dummy);
	fscanf(r1,"%s\n",&dummy);
	fscanf(r1,"%s\n",&dummy);
	fscanf(r1,"%s\n",&dummy);
	fscanf(r1,"%s\n",&dummy);
	fscanf(r1,"%s\n",&dummy);
	fscanf(r1,"%s\n",&dummy);
	fscanf(r1,"%s\n",&dummy);
	fscanf(r2,"%s\n",&dummy);
	fscanf(r2,"%s\n",&dummy);
	fscanf(r2,"%s\n",&dummy);
	fscanf(r2,"%s\n",&dummy);
	fscanf(r2,"%s\n",&dummy);
	fscanf(r2,"%s\n",&dummy);
	fscanf(r2,"%s\n",&dummy);
	fscanf(r2,"%s\n",&dummy);
	fscanf(r2,"%s\n",&dummy);

	for(i=0;i<rp1;i++){
		fscanf(r1,"%d\n",&pi);
		fprintf(w,"%d\n",pi);
		//printf("%d\n",pi);
	}
	for(i=0;i<rp2;i++){
		fscanf(r2,"%d\n",&pi);
		fprintf(w,"%d\n",pi);
	}
	fclose(r1);
	fclose(r2);
	fclose(w);


	for(j=1;j<10;j++){
		sprintf(FileName1,"./plotdata/fluid_%d0000stp-0.vtk",j);
		sprintf(FileName2,"./plotdata/fluid_%d0000stp-1.vtk",j);
		sprintf(FileNamew,"./merge/fluid_%d0000stp.vtk",j);
		r1=fopen(FileName1,"r+");
		r2=fopen(FileName2,"r+");
		w=fopen(FileNamew,"w+");

		fprintf(w,"# vtk DataFile Version 3.0\n");
		fprintf(w,"Print out results in vtk format\n");
		fprintf(w,"ASCII\n");
		fprintf(w,"DATASET POLYDATA\n");

		fscanf(r1,"%d\n",&rp1);
		fscanf(r2,"%d\n",&rp2);
		wp=rp1+rp2;
		fprintf(w,"POINTS\t%d\tfloat\n",wp);					// define particles position as POINTS

		for(i=0;i<rp1;i++){
			fscanf(r1,"%f\t%f\t%f\t\n",&x,&y,&z);
			fprintf(w,"%f\t%f\t%f\t\n",x,y,z);
		}
		for(i=0;i<rp2;i++){
			fscanf(r2,"%f\t%f\t%f\t\n",&x,&y,&z);
			fprintf(w,"%f\t%f\t%f\t\n",x,y,z);
		}
		fprintf(w,"VERTICES\t%d\t%d\n",wp,2*wp);
		for(i=0;i<wp;i++){
			fprintf(w,"1\t%d\n",i);
		}
		fprintf(w,"POINT_DATA\t%d\n",wp);
		fprintf(w,"FIELD FieldData\t1\n");
		fprintf(w,"p-type\t1\t%d\tfloat\n",wp);

		fscanf(r1,"%s\n",&dummy);
		fscanf(r1,"%s\n",&dummy);
		fscanf(r1,"%s\n",&dummy);
		fscanf(r1,"%s\n",&dummy);
		fscanf(r1,"%s\n",&dummy);
		fscanf(r1,"%s\n",&dummy);
		fscanf(r1,"%s\n",&dummy);
		fscanf(r1,"%s\n",&dummy);
		fscanf(r1,"%s\n",&dummy);
		fscanf(r2,"%s\n",&dummy);
		fscanf(r2,"%s\n",&dummy);
		fscanf(r2,"%s\n",&dummy);
		fscanf(r2,"%s\n",&dummy);
		fscanf(r2,"%s\n",&dummy);
		fscanf(r2,"%s\n",&dummy);
		fscanf(r2,"%s\n",&dummy);
		fscanf(r2,"%s\n",&dummy);
		fscanf(r2,"%s\n",&dummy);

		for(i=0;i<rp1;i++){
			fscanf(r1,"%d\n",&pi);
			fprintf(w,"%d\n",pi);
		}
		for(i=0;i<rp2;i++){
			fscanf(r2,"%d\n",&pi);
			fprintf(w,"%d\n",pi);
		}
		fclose(r1);
		fclose(r2);
		fclose(w);
	}
}
//*/
float FloatSwap( float f )
{
   union
   {
      float f;
      unsigned char b[4];
      //unsigned char b[8];
   } dat1,dat2;

   dat1.f=f;
   dat2.b[0]=dat1.b[3];
   dat2.b[1]=dat1.b[2];
   dat2.b[2]=dat1.b[1];
   dat2.b[3]=dat1.b[0];
   return dat2.f;
}
////////////////////////////////////////////////////////////////////////
int IntSwap( int d )
{
   union
   {
      int d;
      unsigned char b[4];
      //unsigned char b[8];
   } dat1,dat2;

   dat1.d=d;
   dat2.b[0]=dat1.b[3];
   dat2.b[1]=dat1.b[2];
   dat2.b[2]=dat1.b[1];
   dat2.b[3]=dat1.b[0];
   return dat2.d;
}
//////////////////////////////////////////////////////////////////////////////
int main(){
	int N_str, freq, Nsim;
	int flag;
 	printf("output starting frame no. output-frequency(interval) number of frames :");
	scanf("%d %d %d",&N_str, &freq, &Nsim);
	printf("Pool particle or Jet particle? (0/1) :");
	scanf("%d",&flag);

	FILE*r1,*r2,*r3,*r4,*r5,*r6;
	FILE*w;
	int i,j,pi;
	char FileName1[256],FileName2[256],FileName3[256],FileName4[256],FileName5[256],FileName6[256],FileNamew[256];
	// char dummy[512];
	int rp1,rp2,rp3,rp4,rp5,rp6,wp;
	// float x,y,z;
	float val;
	int valt;
	float3 bin;
	char ch;

	for(j=0;j<Nsim;j++){
		pi=N_str+j*freq;
		// sprintf(FileName1,"./plotdata/fluid_%dstp-0.vtk",pi);
		// sprintf(FileName2,"./plotdata/fluid_%dstp-1.vtk",pi);
		// VTK
		// sprintf(FileNamew,"./merge/fluid_%dstp.vtk",pi);
		// PLY
		// sprintf(FileNamew,"./merge/jetfluid_%dstp.ply",pi);
		// sprintf(FileNamew,"./merge/poolfluid_%dstp.ply",pi);
		if(flag<1){
			sprintf(FileName1,"./plotdata/poolfluid_%dstp-0.vtk",pi);
			sprintf(FileName2,"./plotdata/poolfluid_%dstp-1.vtk",pi);

			sprintf(FileName3,"./plotdata/poolfluid_%dstp-2.vtk",pi);
			// sprintf(FileName4,"./plotdata/poolfluid_%dstp-3.vtk",pi);
			// sprintf(FileName5,"./plotdata/poolfluid_%dstp-4.vtk",pi);
			// sprintf(FileName6,"./plotdata/poolfluid_%dstp-5.vtk",pi);

			sprintf(FileNamew,"./merge/poolfluid_%dstp.vtk",pi);
		}else{
			sprintf(FileName1,"./plotdata/jetfluid_%dstp-0.vtk",pi);
			sprintf(FileName2,"./plotdata/jetfluid_%dstp-1.vtk",pi);

			sprintf(FileName3,"./plotdata/jetfluid_%dstp-2.vtk",pi);
			// sprintf(FileName4,"./plotdata/jetfluid_%dstp-3.vtk",pi);
			// sprintf(FileName5,"./plotdata/jetfluid_%dstp-4.vtk",pi);
			// sprintf(FileName6,"./plotdata/jetfluid_%dstp-5.vtk",pi);

			sprintf(FileNamew,"./merge/jetfluid_%dstp.vtk",pi);
		}

		r1=fopen(FileName1,"r");
		r2=fopen(FileName2,"r");
    //
		r3=fopen(FileName3,"r");
		// r4=fopen(FileName4,"r");
		// r5=fopen(FileName5,"r");
		// r6=fopen(FileName6,"r");

		w=fopen(FileNamew,"w");

		fprintf(w,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
		fprintf(w,"Print out results in vtk format\n");				// header: description of file, it never exceeds 256 characters
		fprintf(w,"BINARY\n");										// format of data (ACSII / BINARY)
		fprintf(w,"DATASET POLYDATA\n");								// define DATASET format: 'POLYDATA' is proper to represent SPH particles

		fscanf(r1,"%d\n",&rp1);
		fscanf(r2,"%d\n",&rp2);
    //
		fscanf(r3,"%d\n",&rp3);
		// fscanf(r4,"%d\n",&rp4);
		// fscanf(r5,"%d\n",&rp5);
		// fscanf(r6,"%d\n",&rp6);

		// wp=rp1+rp2+rp3+rp4+rp5+rp6;
		wp=rp1+rp2+rp3;
    // wp=rp1;

		fprintf(w,"POINTS\t%d\tfloat\n",wp);						// define particles position as POINTS
		printf("Number of total POINTS:: %d\n",wp);

		// fprintf(w,"ply\n");
		// fprintf(w,"format binary_big_endian 1.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
		// fprintf(w,"comment made by S.H.Park\n");						// header: description of file, it never exceeds 256 characters
		// fprintf(w,"comment test for water jet\n");
		// fprintf(w,"element vertex %d\n", wp);
		// fprintf(w,"property float x\n");
		// fprintf(w,"property float y\n");
		// fprintf(w,"property float z\n");
		// fprintf(w,"property float intensity\n");
		// // fprintf(w,"property uchar diffuse_red\n");
		// // fprintf(w,"property uchar diffuse_green\n");
		// // fprintf(w,"property uchar diffuse_blue\n");
		// fprintf(w,"end_header\n");


		for(i=0;i<rp1;i++){
			fread((void*)&bin,sizeof(float),3,r1);
			fwrite((void*)&bin, sizeof(float),3,w);

			// val=FloatSwap(1);
			// fwrite((void*)&val, sizeof(float), 1, w);
			// ch='255';
			// fwrite(&ch, sizeof(char), 1, w);
			// ch='243';
			// fwrite(&ch, sizeof(char), 1, w);
			// ch='245';
			// fwrite(&ch, sizeof(char), 1, w);
		}
		for(i=0;i<rp2;i++){
			fread((void*)&bin,sizeof(float),3,r2);
			fwrite((void*)&bin, sizeof(float),3,w);

			// val=FloatSwap(1);
			// fwrite((void*)&val, sizeof(float), 1, w);
			// ch='255';
			// fwrite(&ch, sizeof(char), 1, w);
			// ch='243';
			// fwrite(&ch, sizeof(char), 1, w);
			// ch='245';
			// fwrite(&ch, sizeof(char), 1, w);
		}

		for(i=0;i<rp3;i++){
			fread((void*)&bin,sizeof(float),3,r3);
			fwrite((void*)&bin, sizeof(float),3,w);
		}
		// for(i=0;i<rp4;i++){
		// 	fread((void*)&bin,sizeof(float),3,r4);
		// 	fwrite((void*)&bin, sizeof(float),3,w);
		// }
		// for(i=0;i<rp5;i++){
		// 	fread((void*)&bin,sizeof(float),3,r5);
		// 	fwrite((void*)&bin, sizeof(float),3,w);
		// }
		// for(i=0;i<rp6;i++){
		// 	fread((void*)&bin,sizeof(float),3,r6);
		// 	fwrite((void*)&bin, sizeof(float),3,w);
		// }


		fprintf(w,"POINT_DATA\t%d\n",wp);
		// all the data of particles are FieldData except 'index' ( 3: declare number of property data )
		fprintf(w,"FIELD FieldData\t1\n");

		fprintf(w,"pressure\t1\t%d\tfloat\n",wp);

		for(i=0;i<rp1;i++){
			fread((void*)&val,sizeof(float),1,r1);
			fwrite((void*)&val,sizeof(float),1,w);
		}
		for(i=0;i<rp2;i++){
			fread((void*)&val,sizeof(float),1,r2);
			fwrite((void*)&val,sizeof(float),1,w);
		}


		for(i=0;i<rp3;i++){
			fread((void*)&val,sizeof(float),1,r3);
			fwrite((void*)&val,sizeof(float),1,w);
		}
		// for(i=0;i<rp4;i++){
		// 	fread((void*)&val,sizeof(float),1,r4);
		// 	fwrite((void*)&val,sizeof(float),1,w);
		// }
		// for(i=0;i<rp5;i++){
		// 	fread((void*)&val,sizeof(float),1,r5);
		// 	fwrite((void*)&val,sizeof(float),1,w);
		// }
		// for(i=0;i<rp6;i++){
		// 	fread((void*)&val,sizeof(float),1,r6);
		// 	fwrite((void*)&val,sizeof(float),1,w);
		// }


		// fprintf(w,"density\t1\t%d\tfloat\n",wp);
    //
		// for(i=0;i<rp1;i++){
		// 	fread((void*)&val,sizeof(float),1,r1);
		// 	fwrite((void*)&val,sizeof(float),1,w);
		// }
		// for(i=0;i<rp2;i++){
		// 	fread((void*)&val,sizeof(float),1,r2);
		// 	fwrite((void*)&val,sizeof(float),1,w);
		// }
		/*
		for(i=0;i<rp3;i++){
			fread((void*)&val,sizeof(float),1,r3);
			fwrite((void*)&val,sizeof(float),1,w);
		}
		for(i=0;i<rp4;i++){
			fread((void*)&val,sizeof(float),1,r4);
			fwrite((void*)&val,sizeof(float),1,w);
		}
		for(i=0;i<rp5;i++){
			fread((void*)&val,sizeof(float),1,r5);
			fwrite((void*)&val,sizeof(float),1,w);
		}
		for(i=0;i<rp6;i++){
			fread((void*)&val,sizeof(float),1,r6);
			fwrite((void*)&val,sizeof(float),1,w);
		}
		//*/
		fclose(r1);
		fclose(r2);
    //
		fclose(r3);
		// fclose(r4);
		// fclose(r5);
		// fclose(r6);

		fclose(w);
	}
}
