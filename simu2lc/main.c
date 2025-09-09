/*
 *+
 * NAME:
 *   simu2LC
 *
 * PURPOSE:
 *   make light-cone from a given simulation
 *
 * REVISION:
 *   20151102
 *-
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

#ifndef WRITEBOXINFO
 #define WRITEBOXINFO
#endif

#define HubbleDistance 2997.92458

void init_config(void);
float func(float x);
float qromb(float (*func)(float), float a, float b);
double BoxHeaderIO(char *FILENAME);

static void polint(float xa[], float ya[], int n, float x, float *y, float *dy);
static float trapzd(float (*func)(float), float a, float b, int n);
static float *vector(long nl, long nh);
static void nrerror(char error_text[]);
static void free_vector(float *v, long nl, long nh);
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
typedef struct
{
  unsigned int npart[6];
  double mass[6];
  double time;
  double redshift;
  int flag_sfr;
  int flag_feedback;
  unsigned int npartTotal[6];
  int flag_cooling;
  int num_files;
  double BoxSize;
  double Omega0;
  double OmegaLambda;
  double HubbleParam;
  char fill[256-12*sizeof(double)-16*sizeof(int)];
} simuHeader;

typedef struct
{
  int snap;
  float redshift;
  float ComvingDis;
}snapInfo;

typedef struct
{
  int ID;
  int Pos[3];
  int snap;
  int subBoxID;
}boxInfo;

typedef struct
{
  //char path_simu[1024];
  char path_out[1024];

  float MaxComvingDis;
  float boxSize;
  int NumSubBoxCut;  //cut num in each big box
  int NumSimuSnap;

  float param;
  float paraml;
}configInfo;

simuHeader header;
snapInfo   *snapIn;
configInfo configIn;
boxInfo    *box;
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

int main(int argc, char **argv)
{
  //sprintf(configIn.path_simu, "/media/lun8/wcl/C4/3072/divisions");
  // sprintf(configIn.path_out,  "/huawei/osv1/wcl/RAYTRACING/L500/lensplanesII");
  sprintf(configIn.path_out,  "your/path/output");
  configIn.MaxComvingDis = 4000.0;  //in Mpc/h
  configIn.boxSize = 500.0;         //in Mpc/h
  configIn.NumSubBoxCut = 5;        //100 Mpc/h, 125 subboxes for 1 big box

  init_config();
  fprintf(stdout,"DONE!\n");
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

void init_config(void)
{
  int NumSimuBoxSide;
  float LightConeOriginX, LightConeOriginY, LightConeOriginZ;
  float subBoxSize;
  int NumSubBox;
  int NumBox;
  // Light Cone Origin Position
  LightConeOriginX = 35; //configIn.boxSize/4.0;   //O=[250, 250, 250] Mpc/h
  LightConeOriginY = 35; //configIn.boxSize/4.0;
  LightConeOriginZ = 35; //configIn.boxSize/4.0;
  /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
  /*simu-box info*/
  NumSimuBoxSide = (configIn.MaxComvingDis)/configIn.boxSize+1; //7 for 3000 Mpc/h
  // if( NumSimuBoxSide % 2 == 1 ) NumSimuBoxSide += 1; //even number

  // // LightConeOriginX += (NumSimuBoxSide/2)*configIn.boxSize; // +16/2*500 central
  // // LightConeOriginY += (NumSimuBoxSide/2)*configIn.boxSize;
  // // LightConeOriginZ += (NumSimuBoxSide/2)*configIn.boxSize;
  // NumSimuBoxSide += 1; //odd number, for putting the Oring into the central

  /*sub-box info*/
  subBoxSize = configIn.boxSize/configIn.NumSubBoxCut;   // 100 Mpc/h
  NumSubBox = configIn.NumSubBoxCut*configIn.NumSubBoxCut*configIn.NumSubBoxCut; // 125 subboxes

  NumBox = NumSimuBoxSide*NumSimuBoxSide*NumSimuBoxSide*NumSubBox; // 7*7*7*125

  fprintf(stdout,"Debug:\n");
  fprintf(stdout,"BoxSize = %f, MaxComvingDistance = %f|%f\n",\
          configIn.boxSize, configIn.MaxComvingDis, configIn.boxSize*NumSimuBoxSide/2);
  fprintf(stdout,"NumSimuBoxSide = %d, LightConeOrigin(X,Y,Z) = (%f %f %f) \n",\
          NumSimuBoxSide, LightConeOriginX, LightConeOriginY, LightConeOriginZ);
  fprintf(stdout,"NumSubBoxCut = %d, NumSubBox = %ld, subBoxSize = %f, NumBox = %ld\n", \
          configIn.NumSubBoxCut, NumSubBox, subBoxSize, NumBox);
  fprintf(stdout,"\n");
  /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
  
  int i, isnap;
  char buf[1024];
  FILE *fd = NULL;
  unsigned int dummy, d0, d1;

  configIn.NumSimuSnap = 38; //38, 62->99
  snapIn =(snapInfo *)malloc(sizeof(snapInfo)*(configIn.NumSimuSnap));

  #define SKIP fread(&dummy, sizeof(dummy), 1, fd);
  for(i = 0; i<configIn.NumSimuSnap; i++)
  {
	isnap = 99-i;
    snapIn[i].snap = isnap;
    snapIn[i].redshift = 0;
    snapIn[i].ComvingDis = 0;

	//if(snapIn[i].snap < 75) continue;

    //sprintf(buf, "/data/hwraid/sjli/L500_3072/snapdir_%03d/snapshot_%03d.0", isnap, isnap);
	//printf("%s\n",buf);
    //assert( (fd = fopen(buf, "r")) != NULL);
    //SKIP;  d0=dummy;
    //fread(&header,sizeof(header),1,fd );
    //SKIP;  d1=dummy;
    //assert(d0 == d1);
    //fclose(fd);
    //assert( header.BoxSize == configIn.boxSize );
    //configIn.param = header.Omega0;
    //configIn.paraml= header.OmegaLambda;
    //snapIn[i].redshift = header.redshift;

    //
    sprintf(buf,"/path/to/simulation/L500/split100/snapdir_%03d/output_0/box_0", isnap); 

    snapIn[i].redshift = BoxHeaderIO(buf); // Obtain the redshift of the big box

    snapIn[i].ComvingDis = HubbleDistance*qromb(*func,0.0, header.redshift);;

    if(i==0) fprintf(stdout,"[om oml h]=[%f %f %f]\n",
                     configIn.param, configIn.paraml, header.HubbleParam);
    fprintf(stdout,"%d: [z, Dc(Mpc/h), Dc(Mpc)]=[%12.4f %12.4f %12.4f]\n\n",
            snapIn[i].snap, snapIn[i].redshift, snapIn[i].ComvingDis,
            snapIn[i].ComvingDis/header.HubbleParam);
    //snapdir is the big box. box_0 is the first subbox of the big box.
  }
  #undef SKIP
  /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
  int tmpNSide;
  int ibox;
  float BoxRadius, ra, rb;

  box = (boxInfo *)malloc(sizeof(boxInfo)* NumBox);
  tmpNSide = configIn.NumSubBoxCut*NumSimuBoxSide;//5*17, side length
  for(ibox=0; ibox<NumBox; ibox++)
  {
    box[ibox].ID = ibox;
    box[ibox].Pos[2] = ibox/(tmpNSide*tmpNSide); // Integral numer. Z_index of the subbox for whole light cone.
    box[ibox].Pos[1] = (ibox - box[ibox].Pos[2]*tmpNSide*tmpNSide) / tmpNSide;// Integral number. X
    box[ibox].Pos[0] = ibox - box[ibox].Pos[2]*tmpNSide*tmpNSide - box[ibox].Pos[1]*tmpNSide; // Integral number. Y
    // printf("%f\n",(float)ibox - box[ibox].Pos[2]*tmpNSide*tmpNSide - box[ibox].Pos[1]*tmpNSide);
    // fprintf(stdout,"Box(X,Y,Z) = (%d %d %d) \n",\
    //       box[ibox].Pos[2], box[ibox].Pos[1], box[ibox].Pos[0]);

    box[ibox].snap = -1;
    box[ibox].subBoxID = (box[ibox].Pos[2] % configIn.NumSubBoxCut)*configIn.NumSubBoxCut*configIn.NumSubBoxCut \
                        +(box[ibox].Pos[1] % configIn.NumSubBoxCut)*configIn.NumSubBoxCut                       \
                        + box[ibox].Pos[0] % configIn.NumSubBoxCut; // Obtain the subbox number.
    // printf("%f\n",(float)box[ibox].subBoxID);
    BoxRadius = pow( (((float)box[ibox].Pos[0]+0.5)*subBoxSize - LightConeOriginX), 2) \
               +pow( (((float)box[ibox].Pos[1]+0.5)*subBoxSize - LightConeOriginY), 2) \
               +pow( (((float)box[ibox].Pos[2]+0.5)*subBoxSize - LightConeOriginZ), 2);
    BoxRadius = sqrt(BoxRadius);//
    // BoxRadius = 0.0; 
    // printf("%f\n",(float)BoxRadius);
    for(i=0; i<configIn.NumSimuSnap; i++) // Choose which big box(snapdir).
    {  
      if(i == 0)  ra = 0.0;
      if(i != 0)  ra = 0.5*(snapIn[i].ComvingDis + snapIn[i-1].ComvingDis);
      if(i != configIn.NumSimuSnap-1) rb = 0.5*(snapIn[i].ComvingDis + snapIn[i+1].ComvingDis);
      if(i == configIn.NumSimuSnap-1) rb = 4000.0;

      if( ra <= BoxRadius && BoxRadius < rb )
      {
        box[ibox].snap = snapIn[i].snap;
        break;
      }
    }

    box[ibox].Pos[2] = box[ibox].Pos[2]*subBoxSize - LightConeOriginZ;
    box[ibox].Pos[1] = box[ibox].Pos[1]*subBoxSize - LightConeOriginY;
    box[ibox].Pos[0] = box[ibox].Pos[0]*subBoxSize - LightConeOriginX;
    // fprintf(stdout,"%d\n",box[ibox].Pos[2]);
    // fprintf(stdout,"Box(X,Y,Z) = (%f %f %f) \n",\
    //       box[ibox].Pos[2], box[ibox].Pos[1], box[ibox].Pos[0]);
    // fprintf(stdout,"BoxRadius = %f, Box(X,Y,Z) = (%d %d %d) \n",\
    //       BoxRadius, box[ibox].Pos[2], box[ibox].Pos[1], box[ibox].Pos[0]);
  }

  /*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
  #ifdef WRITEBOXINFO
    
    fprintf(stderr, "write to disk...\n");
    FILE *fp;

    // Write to a .bin file.
    sprintf(buf,"%s/boxInfo.bin", configIn.path_out);
    assert( (fp = fopen(buf,"w")) != NULL);

    fwrite(&configIn.boxSize, sizeof(configIn.boxSize),1,fp);
    fwrite(&configIn.NumSubBoxCut, sizeof(configIn.NumSubBoxCut),1,fp);
    fwrite(&NumBox, sizeof(NumBox),1,fp);
    for(ibox=0; ibox<NumBox; ibox++)
    {
      fwrite(&box[ibox].ID, sizeof(box[ibox].ID),1,fp);
      fwrite(&box[ibox].Pos[0],sizeof(box[ibox].Pos[0]),1,fp);
      fwrite(&box[ibox].Pos[1],sizeof(box[ibox].Pos[1]),1,fp);
      fwrite(&box[ibox].Pos[2],sizeof(box[ibox].Pos[2]),1,fp);
      fwrite(&box[ibox].snap, sizeof(box[ibox].snap),1,fp);
      fwrite(&box[ibox].subBoxID, sizeof(box[ibox].subBoxID),1,fp);
    }
    fwrite(&NumBox, sizeof(NumBox),1,fp);
    fclose(fp);
    
    // Write to a .txt file

    char buftxt[1024];
    sprintf(buftxt,"%s/boxInfo.txt", configIn.path_out);
    assert( (fp = fopen(buftxt,"w")) != NULL);
    fprintf(fp,"Boxsize:%.0f,NumSubBoxCut:%d,NumBox:%ld\n",configIn.boxSize,configIn.NumSubBoxCut,NumBox);
    // fprintf(fp,"  ID     Box(X,Y,Z)          snap          subBoxID\n");
    for(ibox=0; ibox<NumBox; ibox++)
    {
      fprintf(fp,"  %d  ",box[ibox].ID);
      fprintf(fp,"(%d %d %d)     ",box[ibox].Pos[2],box[ibox].Pos[1],box[ibox].Pos[0]);
      fprintf(fp,"%d      ",box[ibox].snap);
      fprintf(fp,"%d\n",box[ibox].subBoxID);
    }
    
    fprintf(stdout,"%f",box[5].ID);
    fclose(fp);
  #endif

  free(box);
  free(snapIn);
}
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/

/**********************integration of the func*********************************/
float func(float x)
{ float y;
  y=configIn.param*(1.0+x)*(1.0+x)*(1.0+x)+configIn.paraml;
  y=(float) pow((double) y, (double)(-0.5));
  return y;
}
/******************************************************************************/
#define EPS 1.0e-6
#define JMAX 20
#define JMAXP (JMAX+1)
#define K 5
float qromb(float (*func)(float), float a, float b)
{
	float ss,dss;
	float s[JMAXP],h[JMAXP+1];
	int j;

	h[1]=1.0;
	for (j=1;j<=JMAX;j++) {
		s[j]=trapzd(func,a,b,j);
		if (j >= K) {
			polint(&h[j-K],&s[j-K],K,0.0,&ss,&dss);
			if (fabs(dss) <= EPS*fabs(ss)) return ss;
		}
		h[j+1]=0.25*h[j];
	}
	nrerror("Too many steps in routine qromb");
	return 0.0;
}
#undef EPS
#undef JMAX
#undef JMAXP
#undef K
/******************************************************************************/
static void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}
/******************************************************************************/
#define FUNC(x) ((*func)(x))
static float trapzd(float (*func)(float), float a, float b, int n)
{
	float x,tnm,sum,del;
	static float s;
	int it,j;

	if (n == 1) {
		return (s=0.5*(b-a)*(FUNC(a)+FUNC(b)));
	} else {
		for (it=1,j=1;j<n-1;j++) it <<= 1;
		tnm=it;
		del=(b-a)/tnm;
		x=a+0.5*del;
		for (sum=0.0,j=1;j<=it;j++,x+=del) sum += FUNC(x);
		s=0.5*(s+(b-a)*sum/tnm);
		return s;
	}
}
#undef FUNC
/******************************************************************************/
#define NRANSI
static void polint(float xa[], float ya[], int n, float x, float *y, float *dy)
{
	int i,m,ns=1;
	float den,dif,dift,ho,hp,w;
	float *c,*d;

	dif=fabs(x-xa[1]);
	c=vector(1,n);
	d=vector(1,n);
	for (i=1;i<=n;i++) {
		if ( (dift=fabs(x-xa[i])) < dif) {
			ns=i;
			dif=dift;
		}
		c[i]=ya[i];
		d[i]=ya[i];
	}
	*y=ya[ns--];
	for (m=1;m<n;m++) {
		for (i=1;i<=n-m;i++) {
			ho=xa[i]-x;
			hp=xa[i+m]-x;
			w=c[i+1]-d[i];
			if ( (den=ho-hp) == 0.0) nrerror("Error in routine polint");
			den=w/den;
			d[i]=hp*den;
			c[i]=ho*den;
		}
		*y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
	}
	free_vector(d,1,n);
	free_vector(c,1,n);
}
#undef NRANSI
/******************************************************************************/
#define NR_END 1
#define FREE_ARG char*
static float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
	float *v;

	v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
	if (!v) nrerror("allocation failure in vector()");
	return v-nl+NR_END;
}
#undef NR_END
#undef FREE_ARG
/******************************************************************************/
#define NR_END 1
#define FREE_ARG char*
static void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}
#undef NR_END
#undef FREE_ARG
/******************************************************************************/


double BoxHeaderIO(char *FILENAME)
{
  FILE *fp;
  double redshift;
  assert( (fp=fopen(FILENAME,"r"))!= NULL );

  int i, ThisTask, NTasks, outputSer;
  int dummy;
  float *pos, *hsml;

  fread(&NTasks, sizeof(int), 1, fp);  
  rewind(fp);
  //Ntasks
  for(i=0; i<NTasks; i++)
  {
    fread(&NTasks, sizeof(int), 1, fp);
    if(i==0) fprintf(stderr, "NTasks=%d\t", NTasks);
  }
  //fprintf(stderr,"\n\n");
  fflush(stderr);
  //outputSer
  for(i=0; i<NTasks; i++)
  {
    fread(&outputSer, sizeof(int), 1, fp);
    if(i==0) fprintf(stderr, "outputSer=%d\t", outputSer);
  }
  //fprintf(stderr,"\n\n");
  fflush(stderr);
  //ThisTask
  for(i=0; i<NTasks; i++)
  {
    fread(&ThisTask, sizeof(int), 1, fp);
    if(i==0) fprintf(stderr, "outputSer=%d\t", ThisTask);
  }
  //fprintf(stderr,"\n\n");
  fflush(stderr);
  //header
  for(i=0; i<NTasks; i++)
  {
    fread(&header, sizeof(header), 1, fp);
    if(i==0) fprintf(stderr, "mass=%f\t", header.mass[1]);
  }
  fprintf(stderr,"\n");
  fflush(stderr);
  assert( header.BoxSize == configIn.boxSize );
  configIn.param = header.Omega0;
  configIn.paraml= header.OmegaLambda;

  redshift = header.redshift;
/*
  //dummy
  long npart;
  for(i=0,npart=0; i<NTasks; i++)
  {
    fread(&dummy, sizeof(int), 1, fp);
    fprintf(stderr, "%d\t", dummy);
    npart += dummy;
  }
  fprintf(stderr,"\n\n");
  fflush(stderr);
  pos = (float *)malloc(sizeof(float)*3*npart);
  hsml= (float *)malloc(sizeof(float)*npart);
  //pos
  fread(pos, sizeof(float), 3*npart, fp);
  //hsml
  fread(hsml, sizeof(float), npart, fp);
  for(i=0; i<npart; i++)
    assert(hsml[i]>=0 && hsml[i] <=10);
  //dummy
  for(i=0,npart=0; i<NTasks; i++)
  {
    fread(&dummy, sizeof(int), 1, fp);
    fprintf(stderr, "%d\t", dummy);
    npart += dummy;
  }
  fprintf(stderr,"\n\n");
  fflush(stderr);
  free(pos);
  free(hsml);
*/
  fclose(fp);
  return redshift;
}
