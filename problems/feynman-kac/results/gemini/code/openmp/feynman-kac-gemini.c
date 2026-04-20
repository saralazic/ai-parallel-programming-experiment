# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>

int main ( int argc, char **argv );
double potential ( double a, double x );
double r8_uniform_01 ( int *seed );
void timestamp ( );


int main ( int argc, char **argv )
{
  double a = 2.0;
  double chk;
  double dx;
  double err;
  double h = 0.0001;
  int i;
  int it;
  int j;
  int k;
  int n1;
  int n = 10000;
  int n_int;
  int ni;
  double rth;
  int seed = 123456789;
  int steps;
  int steps_ave;
  double sum;
  double test;
  double us;
  double vh;
  double vs;
  double x;
  double x1;
  double w;
  double w_exact;
  double we;
  double wt;

  timestamp ( );

  printf ( "\n" );
  printf ( "FEYNMAN_KAC_1D:\n" );
  printf ( "  C version.\n" );
  printf ( "\n" );
  printf ( "  Program parameters:\n" );
  printf ( "\n" );
  printf ( "  The calculation takes place inside an interval.\n" );
  printf ( "  The solution will be estimated at points\n" );
  printf ( "  on a regular spaced grid within the interval.\n" );
  printf ( "  Each solution will be estimated by computing %d trajectories\n", n );
  printf ( "  from the point to the boundary.\n" );
  printf ( "\n" );
  printf ( "    (X/A)^2 = 1\n" );
  printf ( "\n" );
  printf ( "  The interval parameter A is:\n" );
  printf ( "\n" );
  printf ( "    A = %g\n", a );
  printf ( "\n" );
  printf ( "  Path stepsize H = %g\n", h );
/*
  Choose the spacing so we have about ni points on or in the interval.
*/
  ni = 21;

  printf ( "\n" );
  printf ( "  X coordinate discretized by %d points\n", ni + 2 );
/*
  RTH is the scaled stepsize.
*/
  rth = sqrt ( h );

  err = 0.0;
/*
  Loop over the points.
*/
  printf ( "\n" );
  printf ( "     I     K       X           W exact" );
  printf ( "      W Approx        Error      Ave Steps  Test\n" );
  printf ( "\n" );

  k = 0;
  n_int = 0;

  for ( i = 0; i <= ni + 1; i++ )
  {
    x = ( ( double ) ( ni - i     ) * ( - a ) 
        + ( double ) (      i - 1 ) *     a ) 
        / ( double ) ( ni     - 1 );

    k = k + 1;

    test = a * a - x * x;

    if ( test < 0.0 )
    {
      w_exact = 1.0;
      wt = 1.0;
      steps_ave = 0;
      printf ( "  %4d  %4d  %12g  %12g  %12g  %12g  %8d  %8g\n",
        i, k, x, w_exact, wt, fabs ( w_exact - wt ), steps_ave, test );
      continue;
    }

    n_int = n_int + 1;
/*
  Compute the exact solution at this point (x,y,z).
*/
    w_exact = exp ( pow ( x / a, 2 ) - 1.0 );
/*
  Now try to estimate the solution at this point.
*/
    wt = 0.0;
    steps = 0;

#pragma omp parallel for private(x1, w, chk, us, dx, vs, vh, we) reduction(+:wt, steps)
    for ( it = 1; it <= n; it++ )
    {
      int my_seed = seed + i * n + it;

      x1 = x;
/* 
  W = exp(-int(s=0..t) v(X)ds) 
*/
      w = 1.0;
/*
  CHK is < 1.0 while the point is inside the interval.
*/
      chk = 0.0;

      while ( chk < 1.0 )
      {
/*
  Determine DX.
*/
        us = r8_uniform_01 ( &my_seed ) - 0.5;
        if ( us < 0.0 )
        {
          dx = - rth;
        }
        else
        {
          dx = + rth;
        }

        vs = potential ( a, x1 );
/*
  Move to the new point.
*/
        x1 = x1 + dx;

        steps = steps + 1;

        vh = potential ( a, x1 );

        we = ( 1.0 - h * vs ) * w;
        w = w - 0.5 * h * ( vh * we + vs * w );

        chk = pow ( x1 / a, 2 );
      }
      wt = wt + w;
    }
/*
   WT is the average of the sum of the different trials.
*/
    wt = wt / ( double ) ( n );
    steps_ave = steps / n;
/*
  Add error in WT to the running L2 error in the solution.
*/
    err = err + pow ( w_exact - wt, 2 );

    printf ( "  %4d  %4d  %12g  %12g  %12g  %12g  %8d  %8g\n",
      i, k, x, w_exact, wt, fabs ( w_exact - wt ), steps_ave, test );
  }
/*
  Compute the RMS error for all the points.
*/
  err = sqrt ( err / ( double ) ( n_int ) );

  printf ( "\n" );
  printf ( "  RMS absolute error in solution = %g\n", err );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "FEYNMAN_KAC_1D:\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}
/******************************************************************************/

double potential ( double a, double x )
{
  double value;

  value = 2.0 * pow ( x / a / a, 2 ) + 1.0 / a / a;

  return value;
}
/******************************************************************************/

double r8_uniform_01 ( int *seed )
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }
/*
  Although SEED can be represented exactly as a 32 bit integer,
  it generally cannot be represented exactly as a 32 bit real number!
*/
  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}
/******************************************************************************/

void timestamp ( void )
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}