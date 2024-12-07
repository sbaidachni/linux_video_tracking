#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdexcept>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>

#include <syslog.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <errno.h>
#include <mqueue.h>

#include <signal.h>
#include <iostream>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <sys/mman.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define HRES 640
#define VRES 480

#define USEC_PER_MSEC (1000)
#define NANOSEC_PER_MSEC (1000000)
#define NANOSEC_PER_SEC (1000000000)
#define NUM_CPU_CORES (4)
#define TRUE (1)
#define FALSE (0)

#define NUM_THREADS (3)

#define SKIP_ITERATIONS (1)
#define IMAGE_ITERATIONS (100)
#define RATE (100)

cv::Mat buffer_img[30];

#define MY_CLOCK_TYPE CLOCK_MONOTONIC_RAW

int abortTest=FALSE;
int abortS1=FALSE;
int abortS2=FALSE;
int abortS3=FALSE;

// semaphore structures
sem_t semS1;
sem_t semS2;
sem_t semS3;
struct timespec start_time_val;
double start_realtime;
unsigned long long sequencePeriods;

static timer_t timer_1;
static struct itimerspec itime = {{1,0}, {1,0}};
static struct itimerspec last_itime;

static unsigned long long seqCnt=0;

// message queue constants 
#define SNDRCV_MQ "/init_img_mq_2"
#define SNDRCV_MQ_F "/init_img_mq_2_f"
#define MAX_MSG_SIZE 256
#define ERROR (-1)

// message queue attributes
struct mq_attr mq_attr;

struct init_img_message 
{
  cv::Mat currframe;
  int idx;
};

typedef struct
{
    int threadIdx;
} threadParams_t;

// threads' orchestrator
void Sequencer(int id);

// capture images from camera
void *Service_1(void *threadp);

// pick the best frame
void *Service_2(void *threadp);

// save the best frame
void *Service_3(void *threadp);

// utility methods
double getTimeMsec(void);
double realtime(struct timespec *tsptr);
void print_scheduler(void);


// only works for x64 with proper privileges - has worked in past, but new security measures may
// prevent use
static inline unsigned long long tsc_read(void)
{
    unsigned int lo, hi;

    // RDTSC copies contents of 64-bit TSC into EDX:EAX
    asm volatile("rdtsc" : "=a" (lo), "=d" (hi));
    return (unsigned long long)hi << 32 | lo;
}

// not able to read unless enabled by kernel module
static inline unsigned ccnt_read (void)
{
    unsigned cc;
    asm volatile ("mrc p15, 0, %0, c15, c12, 1" : "=r" (cc));
    return cc;
}

int main(void)
{
    struct timespec current_time_val, current_time_res;
    double current_realtime, current_realtime_res;

    int i, rc, scope, flags=0;

    cpu_set_t threadcpu;
    cpu_set_t allcpuset;

    pthread_t threads[NUM_THREADS];
    threadParams_t threadParams[NUM_THREADS];
    pthread_attr_t rt_sched_attr[NUM_THREADS];
    int rt_max_prio, rt_min_prio, cpuidx;

    struct sched_param rt_param[NUM_THREADS];
    struct sched_param main_param;

    pthread_attr_t main_attr;
    pid_t mainpid;

    printf("Starting High Rate Sequencer Demo\n");
    clock_gettime(MY_CLOCK_TYPE, &start_time_val); start_realtime=realtime(&start_time_val);
    clock_gettime(MY_CLOCK_TYPE, &current_time_val); current_realtime=realtime(&current_time_val);
    clock_getres(MY_CLOCK_TYPE, &current_time_res); current_realtime_res=realtime(&current_time_res);
    printf("START High Rate Sequencer @ sec=%6.9lf with resolution %6.9lf\n", (current_realtime - start_realtime), current_realtime_res);

   //timestamp = ccnt_read();
   //printf("timestamp=%u\n", timestamp);

    //message queue attributes
    mq_attr.mq_maxmsg = 50;
    mq_attr.mq_msgsize = MAX_MSG_SIZE;
    mq_attr.mq_flags = 0;

    printf("System has %d processors configured and %d available.\n", get_nprocs_conf(), get_nprocs());

    CPU_ZERO(&allcpuset);

    for(i=0; i < NUM_CPU_CORES; i++)
        CPU_SET(i, &allcpuset);

    // printf("Using CPUS=%d from total available.\n", CPU_COUNT(&allcpuset));


    // initialize the sequencer semaphores
    if (sem_init (&semS1, 0, 0)) { printf ("Failed to initialize S1 semaphore\n"); exit (-1); }
    if (sem_init (&semS2, 0, 0)) { printf ("Failed to initialize S1 semaphore\n"); exit (-1); }
    if (sem_init (&semS3, 0, 0)) { printf ("Failed to initialize S1 semaphore\n"); exit (-1); }
    mainpid=getpid();

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    // apply FIFO to the main thread
    rc=sched_getparam(mainpid, &main_param);
    main_param.sched_priority=rt_max_prio;
    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0) perror("main_param");
    print_scheduler();

    pthread_attr_getscope(&main_attr, &scope);

    if(scope == PTHREAD_SCOPE_SYSTEM)
      printf("PTHREAD SCOPE SYSTEM\n");
    else if (scope == PTHREAD_SCOPE_PROCESS)
      printf("PTHREAD SCOPE PROCESS\n");
    else
      printf("PTHREAD SCOPE UNKNOWN\n");

    printf("rt_max_prio=%d\n", rt_max_prio);
    printf("rt_min_prio=%d\n", rt_min_prio);


    // initialize attributes for threads using core #3
    for(i=0; i < NUM_THREADS; i++)
    {

        CPU_ZERO(&threadcpu);
        cpuidx=(1+i);
        CPU_SET(cpuidx, &threadcpu);

        rc=pthread_attr_init(&rt_sched_attr[i]);
        rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
        rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
        rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);

        rt_param[i].sched_priority=rt_max_prio-i;
        pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);

        threadParams[i].threadIdx=i;
    }
   
    // printf("Service threads will run on %d CPU cores\n", CPU_COUNT(&threadcpu));

    // Create Service threads which will block awaiting release for:

    // Servcie_1 create
    //
    rt_param[0].sched_priority=rt_max_prio-1;
    pthread_attr_setschedparam(&rt_sched_attr[0], &rt_param[0]);
    rc=pthread_create(&threads[0],               // pointer to thread descriptor
                      &rt_sched_attr[0],         // use specific attributes
                      //(void *)0,               // default attributes
                      Service_1,                 // thread function entry point
                      (void *)&(threadParams[0]) // parameters to pass in
                     );
    if(rc < 0)
        perror("pthread_create for service 1");
    else
        printf("pthread_create successful for service 1\n");


    // Service_2 create
    //
    rt_param[1].sched_priority=rt_max_prio-2;
    pthread_attr_setschedparam(&rt_sched_attr[1], &rt_param[1]);
    rc=pthread_create(&threads[1], &rt_sched_attr[1], Service_2, (void *)&(threadParams[1]));
    if(rc < 0)
        perror("pthread_create for service 2");
    else
        printf("pthread_create successful for service 2\n");


    // Service_3 create
    //
    rt_param[1].sched_priority=rt_max_prio-3;
    pthread_attr_setschedparam(&rt_sched_attr[2], &rt_param[2]);
    rc=pthread_create(&threads[2], &rt_sched_attr[2], Service_3, (void *)&(threadParams[2]));
    if(rc < 0)
        perror("pthread_create for service 3");
    else
        printf("pthread_create successful for service 3\n");

    // Wait for service threads to initialize and await relese by sequencer.
    //
    // Note that the sleep is not necessary of RT service threads are created with 
    // correct POSIX SCHED_FIFO priorities compared to non-RT priority of this main
    // program.
    //
    // sleep(1);
 
    // Create Sequencer thread, which like a cyclic executive, is highest prio
    printf("Start sequencer\n");

    sequencePeriods=(IMAGE_ITERATIONS+SKIP_ITERATIONS)*RATE;

    // Sequencer = RT_MAX	@ 100 Hz
    //
    /* set up to signal SIGALRM if timer expires */
    timer_create(CLOCK_REALTIME, NULL, &timer_1);

    signal(SIGALRM, (void(*)(int)) Sequencer);


    /* arm the interval timer */
    itime.it_interval.tv_sec = 0;
    itime.it_interval.tv_nsec = 10000000;
    itime.it_value.tv_sec = 0;
    itime.it_value.tv_nsec = 10000000;
    //itime.it_interval.tv_sec = 1;
    //itime.it_interval.tv_nsec = 0;
    //itime.it_value.tv_sec = 1;
    //itime.it_value.tv_nsec = 0;

    timer_settime(timer_1, flags, &itime, &last_itime);


    for(i=0;i<NUM_THREADS;i++)
    {
        if((rc=pthread_join(threads[i], NULL)) < 0)
		perror("main pthread_join");
	else
		printf("joined thread %d\n", i);
    }

    mq_unlink(SNDRCV_MQ);
    mq_unlink(SNDRCV_MQ_F);

   printf("\nTEST COMPLETE\n");

   return 0;
}

void Sequencer(int id)
{
    int flags=0;

    // received interval timer signal
           
    seqCnt++;

    //clock_gettime(MY_CLOCK_TYPE, &current_time_val); current_realtime=realtime(&current_time_val);
    //printf("Sequencer on core %d for cycle %llu @ sec=%6.9lf\n", sched_getcpu(), seqCnt, current_realtime-start_realtime);
    //syslog(LOG_CRIT, "Sequencer on core %d for cycle %llu @ sec=%6.9lf\n", sched_getcpu(), seqCnt, current_realtime-start_realtime);


    // Release each service at a sub-rate of the generic sequencer rate

    // Servcie_1 = RT_MAX-1
    if((seqCnt % RATE) == 0) sem_post(&semS1);

    if((seqCnt % RATE) == 0) sem_post(&semS2);

    if((seqCnt % RATE) == 0) sem_post(&semS3);

    if(abortTest || (seqCnt >= sequencePeriods))
    {
        abortS1=TRUE;

	    // shutdown all services
        sem_post(&semS1);
    }

    if(abortTest || (seqCnt >= sequencePeriods+3*RATE))
    {
        // disable interval timer
        itime.it_interval.tv_sec = 0;
        itime.it_interval.tv_nsec = 0;
        itime.it_value.tv_sec = 0;
        itime.it_value.tv_nsec = 0;
        timer_settime(timer_1, flags, &itime, &last_itime);
	    printf("Disabling sequencer interval timer with abort=%d and %llu of %lld\n", abortTest, seqCnt, sequencePeriods);
        sem_post(&semS2);
        sem_post(&semS3);
        abortS2=TRUE;
        abortS3=TRUE;
    }
}

void *Service_1(void *threadp)
{
    struct timespec current_time_val;
    double current_realtime;
    unsigned long long S1Cnt=0;
    mqd_t mymq;
    int nbytes;
    char time_text[20];
    int buffer_idx=0;

    cv::VideoCapture camera(0);
    if (!camera.isOpened()) 
    {
        std::cerr << "ERROR: Could not open camera" << std::endl;
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, HRES);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, VRES);


    sem_wait(&semS1);

    do // check for synchronous abort request
    {
        sem_wait(&semS1);
        if (buffer_idx==30)
            buffer_idx=0;
        S1Cnt++;

        mymq = mq_open(SNDRCV_MQ, O_CREAT|O_RDWR, S_IRWXU, &mq_attr);

        if(mymq < 0)
        {
            perror("sender mq_open");
            exit(-1);
        }
        cv::Mat frame;

        camera >> frame;
        if (frame.empty()) 
        {
            std::cerr << "ERROR: Could not grab a frame" << std::endl;
        }

        if (S1Cnt>SKIP_ITERATIONS)
        {
            struct init_img_message msg;

            clock_gettime(MY_CLOCK_TYPE, &current_time_val); current_realtime=realtime(&current_time_val);

            buffer_img[buffer_idx] = frame;

            sprintf(time_text, "%6.3lf",  current_realtime-start_realtime);

            msg.currframe = buffer_img[buffer_idx]; //currframe;
            msg.idx = buffer_idx;
            buffer_idx++;

            if((nbytes = mq_send(mymq, (const char *)&msg, sizeof (struct init_img_message), 30)) == ERROR)
            {
                perror("mq_send");
            }
        
        }
        mq_close (mymq);
    }
    while(!abortS1);

    camera.release();

    pthread_exit((void *)0);
}

void *Service_2(void *threadp)
{
    unsigned long long S2Cnt=0;

    mqd_t mymq;
    char buffer[MAX_MSG_SIZE];
    unsigned int prio;
    int nbytes;

    cv::Mat cf;

    do // check for synchronous abort request
    {
        sem_wait(&semS2);

        S2Cnt++;

        mymq = mq_open(SNDRCV_MQ, O_CREAT|O_RDWR|O_NONBLOCK, S_IRWXU, &mq_attr);

        // check if we have images in the queue
        while ((nbytes = mq_receive (mymq, buffer, MAX_MSG_SIZE, &prio)) !=-1)
        {
            struct init_img_message *msg = (struct init_img_message *)buffer;
            //
            cf = msg->currframe;

            mq_close (mymq);

            Mat gray_img;
            cvtColor(cf, gray_img, COLOR_BGR2GRAY);

            Point topLeft(gray_img.cols / 3, gray_img.rows / 3);
            Point bottomRight(2 * gray_img.cols / 3, 2 * gray_img.rows / 3);

            // Draw the rectangle (color: blue, thickness: 2)
            rectangle(gray_img, topLeft, bottomRight, Scalar(255, 0, 0), 2);

            buffer_img[msg->idx] = gray_img;

            mymq = mq_open(SNDRCV_MQ_F, O_CREAT|O_RDWR, S_IRWXU, &mq_attr);

            if(mymq < 0)
            {
                perror("sender mq_open");
                exit(-1);
            }

            struct init_img_message msg_out;

            msg_out.currframe = gray_img;

            if((nbytes = mq_send(mymq, (const char *)&msg_out, sizeof (struct init_img_message), 30)) == ERROR)
            {
                perror("mq_send");
            }

            mq_close (mymq);
        }

    }
    while(!abortS2);

    // Resource shutdown here
    //
    pthread_exit((void *)0);
}

void *Service_3(void *threadp)
{
    unsigned long long S3Cnt=0;
    char filename[100];

    mqd_t mymq;
    unsigned int prio;
    char buffer[MAX_MSG_SIZE];
    int nbytes;
    int count_image = 0;

    // Open the frame buffer device
    int fb_fd = open("/dev/fb0", O_RDWR);
    if (fb_fd == -1) {
        std::cerr << "Error: Could not open frame buffer device!" << std::endl;
    }

    // Get screen information
    struct fb_var_screeninfo vinfo;
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo)) {
        std::cerr << "Error: Could not get screen information!" << std::endl;
        close(fb_fd);
    }

    // Map the frame buffer to memory
    size_t screensize = vinfo.yres_virtual * vinfo.xres_virtual * vinfo.bits_per_pixel / 8;
    uint8_t* fb_ptr = (uint8_t*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if (fb_ptr == MAP_FAILED) {
        std::cerr << "Error: Could not map frame buffer to memory!" << std::endl;
        close(fb_fd);
    }

    std::cout<<vinfo.yres_virtual<<"\n"<<vinfo.xres_virtual;

    do // check for synchronous abort request
    {
        sem_wait(&semS3);

        S3Cnt++;

        mymq = mq_open(SNDRCV_MQ_F, O_CREAT|O_RDWR|O_NONBLOCK, S_IRWXU, &mq_attr);

        // check if we have images in the queue
        while (((nbytes = mq_receive (mymq, buffer, MAX_MSG_SIZE, &prio)) !=-1))
        {
            struct init_img_message *msg = (struct init_img_message *)buffer;

            // TO TEST ONLY ON a LOCAL
            count_image++;
            sprintf(filename, "img_%d.jpg", count_image);
            if (!cv::imwrite(filename, msg->currframe)) 
            {
                std::cerr << "Error: Could not save image to file" << std::endl;
            }
            // END TEST BLOCK

            try
            {
                // Resize the image to fit the screen
                //cv::resize(msg->currframe, msg->currframe, cv::Size(vinfo.xres, vinfo.yres));
                //memcpy(fb_ptr, (msg->currframe).data, screensize);

                //if we need to copy the image as is
                int width = msg->currframe.cols; 
                int height = msg->currframe.rows; 
                int channels = msg->currframe.channels(); 
                std::cout << "Width: " << width << std::endl;
                std::cout << "Height: " << height << std::endl;
                std::cout << "Channels: "<<channels<<std::endl;

                // Copy the image data to the frame buffer
                memcpy(fb_ptr, (msg->currframe).data, width*height*channels);
            }
            catch (const exception& e) 
            {
                // print the exception
                cout << "Exception " << e.what() << endl;
            }

        }
        mq_close (mymq);
    }
    while(!abortS3);

    // Unmap and close the frame buffer
    munmap(fb_ptr, screensize);
    close(fb_fd);

    // Resource shutdown here
    //
    pthread_exit((void *)0);
}


double getTimeMsec(void)
{
  struct timespec event_ts = {0, 0};

  clock_gettime(MY_CLOCK_TYPE, &event_ts);
  return ((event_ts.tv_sec)*1000.0) + ((event_ts.tv_nsec)/1000000.0);
}

double realtime(struct timespec *tsptr)
{
    return ((double)(tsptr->tv_sec) + (((double)tsptr->tv_nsec)/1000000000.0));
}

void print_scheduler(void)
{
   int schedType;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
       case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
       case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n"); exit(-1);
         break;
       case SCHED_RR:
           printf("Pthread Policy is SCHED_RR\n"); exit(-1);
           break;
       //case SCHED_DEADLINE:
       //    printf("Pthread Policy is SCHED_DEADLINE\n"); exit(-1);
       //    break;
       default:
           printf("Pthread Policy is UNKNOWN\n"); exit(-1);
   }
}
