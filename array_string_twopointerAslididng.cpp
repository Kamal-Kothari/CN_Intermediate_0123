//CN Intermediate 

//1 Time Complexity
1 Linear 
log n Logarithmic
n linear
nlogn linear Logarithmic
n^2 quadratic
n^3 cubic
2^n Exponential
n! factorial

worst big_O , best omega , average theta

//2 Recurrence Relation
T(n)=T(n-1)+1 -> O(n)	//decreasing since n becomes n-1
T(n)=T(n-1)+n -> O(n^2)
T(n)=T(n-1)+log n -> O(nlogn)

//3 Masters Theorem for decreasing function
T(n)=aT(n-b)+f(n)
a>0 b>0 f(n)=O(n^k) k>=0

if a<1
T(n)=f(n)

a=1
T(n)=n*f(n)

a>1 
T(n)=O[f(n)*(a^(n/b))]

//4 Masters Theorem for dividing function
T(n)=aT(n/b)+f(n)
a>=1 b>1 f(n)=O[(n^k)(logn^p)]

1:log a base b > k
then O(n^val) val=log a base b

2:log a base b = k
2.1 p>-1 O[(n^k)(logn^[p+1])]
2.2 p=-1 O[(n^k)(loglogn)]
2.3 p<-1 O[(n^k)]

3:log a base b < k
3.1 p>=0 O[(n^k)(logn^p)]
3.2 p<0 O[(n^k)]

//5 Kadane Algo to find max sum subarray in TC:O(n)
vector <int> v(n);
for(int i=0;i<n;i++)
{
	cin>>v[i];
}

int maxsum=0;//for empty subarray
int sum=0;
for(int val:v)
{
	sum=max(sum+val,0);//continue previous sum if >0 or 
    //start new subarray with sum=0
	maxsum=max(maxsum,sum);
}
cout<<maxsum;

//6 flip bits of a subset to get max 1's
int flipBits(int* arr, int n) 
{
    // WRITE YOUR CODE HERE
    int max=0;
    int sum=0;
    int total=0
    for(int i=0;i<n;i++)
    {
    	if(arr[i])
    	{
    		sum=max(sum-1,0);
    		total++;
    	}
    	else 
    	{
    		sum=max(sum+1,0);
    	}
    	max=max(max,sum);
    }
    return max+total;
}

//7 k concat max sum subarray
//a1
#include <bits/stdc++.h> 
#define ll long long
long long maxSubSumKConcat(vector<int> &arr, int n, int k)
{
    // Write your code here.
    ll maxsub=INT_MIN,csum=0;//non empty
    for(int val:arr)
    {
        //csum=max(csum+val,0ll);
        maxsub=max(maxsub,csum+val);
        csum=max(csum+val,0ll);
    }

    if(k==1) return maxsub;

    ll ans,ss=0,ps=0,sum=0,mss=INT_MIN,mps=INT_MIN;
    for(int val:arr)
    {
        ps+=val;
        mps=max(mps,ps);
    }
    sum=ps;
    for(int i=n-1;i>=0;i--)
    {
        ss+=arr[i];
        mss=max(mss,ss);
    }

    if(sum>0)
    {
        ans=max(mss+(k-2)*sum+mps,maxsub);
    }
    else 
    {
        ans=max(mss+mps,maxsub);
    }
    return ans;
}
//a2
/*
    Time Complexity: O(N)
    Space Complexity: O(1)

    where 'N' is the size of vector/list 'ARR'.
*/


long long kadane(vector<int> &arr, int n, int k)
{
    long long maxSum = -1e15;
    long long curSum = 0; 
  
    for (int i = 0; i < n * k; i++) 
    { 
        curSum += arr[i % n];   
        maxSum = max(maxSum, curSum);  
        if (curSum < 0)
        {
            curSum = 0;
        } 
    } 

    return maxSum;
}


long long maxSubSumKConcat(vector<int> &arr, int n, int k)
{   
    long long maxSubSum;

    if (k == 1)
    {
        maxSubSum = kadane(arr, n, k);

        return maxSubSum;
    }

    int arrSum = 0;

    for (int i = 0; i < n; i++)
    {
        arrSum += arr[i];
    }

    maxSubSum = kadane(arr, n, 2);   
    if (arrSum > 0)
    {   
        maxSubSum += (long long)(k - 2) * (long long)arrSum;
    }

    return maxSubSum;
}

//8 max sum rectangle in 2d matrix
//tc row*row*col
#include <bits/stdc++.h> 
int kadanef(vector<int> &v,int c)
{
    int mss=INT_MIN;
    int csum=0;
    for(int val:v)
    {
        csum+=val;
        mss=max(mss,csum);
        if(csum<0) csum=0;
    }
    return mss;
}

int maxSumRectangle(vector<vector<int>>& arr, int n, int m)
{
    // write your code here
    int res=INT_MIN;
    for(int i=0;i<n;i++)
    {
        vector<int> v(m);
        for(int j=i;j<n;j++)
        {
            for(int col=0;col<m;col++)
            {
                v[col]+=arr[j][col];
            }
            int maxs=kadanef(v,m);
            res=max(res,maxs);
        }
    }
    return res;
}

//9 quick sort using partition
/*
    Time Complexity: O(N * log(N))
    Space Complexity: O(log N)

    Where 'N' is the number of elements in the given array/list.
*/

// An auxilliary function to partition the array/list based on a pivot.
void partition(vector<int> &arr, int beg, int last, int &start, int &mid)
{
    int pivot = arr[last];
    int end = last;
    
    // Iterate while mid is not greater than end.
    while (mid <= end)
    {
        // Place the element at the starting if it's value is less than pivot.
        if (arr[mid] < pivot)
        {
            swap(arr[mid], arr[start]);
            mid = mid + 1;
            start = start + 1;
        }

        // Place the element at the end if it's value is greater than pivot. 
        else if (arr[mid] > pivot)
        {
            swap(arr[mid], arr[end]);
            end = end - 1;
        }

        else
        {
            mid = mid + 1;
        }
    }
}

// An auxiallary function to sort the array.
void quicksort(vector<int> &arr, int beg, int last)
{
    // Base case when the size of array is 1.
    if (beg >= last)
    {
        return;
    }

    // To handle the case when there are only 2 elements in the array.
    if (last == beg + 1)
    {
        if (arr[beg] > arr[last])
        {
            swap(arr[beg], arr[last]);
            return;
        }
    }


    int start = beg, mid = beg;


    // Function to partition the array.
    partition(arr, beg, last, start, mid);

    // Recursively sort the left and the right partitions.
    quicksort(arr, beg, start - 1);
    quicksort(arr, mid, last);

}

vector<int> quickSortUsingDutchNationalFlag(vector<int> &arr)
{
    // Call the quicksort function.
    quicksort(arr, 0, arr.size() - 1);

    // Return the array/list after sorting.
    return arr;
}

//a2
int partition(vector<int> &v,int li,int ri)
{
    int pivot=v[ri];
    int pi=li-1;
    for(int i=li;i<ri;i++)
    {
        if(v[i]<pivot)
        {
            pi++;
            swap(v[i],v[pi]);
        }
    }
    swap(v[ri],v[pi+1]);
    return pi+1;
}

void quicksort(vector<int> &v,int l,int r)
{
    if(l>=r) return;
    int pi=partition(v,l,r);
    quicksort(v,l,pi-1);
    quicksort(v,pi+1,r);
}

int main()
{
    vector<int> v{3,2,5,1,10,0};
    quicksort(v,0,5);
    for(auto i:v) cout<<i<<" ";
    return 0;
}

//10 Search in sorted rotated array
int search(int* arr, int n, int key) {
    // Write your code here.
    int s=0,e=n-1,mid=0;
    while(s<=e)
    {
        mid=(s+e)/2;
        if(arr[mid]==key) return mid;
        else if(arr[mid]>=arr[s])
        {
            if(key>=arr[s] && key<arr[mid]) e=mid-1;
            else s=mid+1;
        }
        else 
        {
            if(key>arr[mid] && key<=arr[e]) s=mid+1;
            else e=mid-1;
        }
    }
    return -1;
}

//11 is possible to make triangle
#include <bits/stdc++.h> 
bool possibleToMakeTriangle(vector<int> &arr)
{
    // Write your code here.
    sort(arr.begin(),arr.end());
    int l,r;
    for(int i=arr.size()-1;i>=2;i--)
    {
        l=0;
        r=i-1;
        while(l<r)
        {
            if(arr[l]+arr[r]>arr[i]) return true;//to count add r-l
            else l++;
        }
    }
    return false;
}

//a2
bool possibleToMakeTriangle(vector<int> &arr)
{
    int n=arr.size();
    if(n<3) return false;
    // Write your code here.
    sort(arr.begin(),arr.end());
    for(int i=n-3;i>=0;i--)//for(int i=0;i<n-2;i++)
    {
        if(arr[i]+arr[i+1]>arr[i+2]) return true;
    }
    return false;
}

//12 first and last occurence
pair<int, int> findFirstLastPosition(vector<int> &arr, int n, int x)
{
	// Write your code here.
    pair<int,int> p(-1,-1);
    int s=0,e=n-1,mid;
    while(s<=e)
    {
        mid=(s+e)/2;
        if(arr[mid]==x) 
        {
            p.first=mid;
            e=mid-1;
        }
        else if(arr[mid]>x) e=mid-1;
        else s=mid+1;
    }
    
    if(p.first==-1) return p;
    
    s=p.first,e=n-1;
    while(s<=e)
    {
        mid=(s+e)/2;
        if(arr[mid]==x) 
        {
            p.second=mid;
            s=mid+1;
        }
        else if(arr[mid]>x) e=mid-1;
        else s=mid+1;
    }
    return p;
}

//13 Count smaller or equal elements in another array
#include <bits/stdc++.h> 
vector < int > countSmallerOrEqual(int * a, int * b, int n, int m) {
    //  Write your code here    
    sort(b,b+m);
    int idx=0;
    for(int i=0;i<n;i++)
    {
        idx=upper_bound(b,b+m,a[i])-b;
        a[i]=idx;
    }
    vector<int> ans(a,a+n);
    return ans;
}

//14 Best insert position 
int bestInsertPos(vector<int> arr, int n, int m)
{
    // Write your code here.
    int idx;
    idx=lower_bound(arr.begin(),arr.end(),m)-arr.begin();
    return idx;
}

//15 Xor query
#include <bits/stdc++.h> 
vector<int> xorQuery(vector<vector<int>> &queries)
{
    // Write your code here
    vector<int> v;
    for(int i=0;i<queries.size();i++)
    {
        if(queries[i][0]==1)
        {
            v.push_back(queries[i][1]);
        }
        else 
        {
            int e=queries[i][1];
            for(int i=0;i<v.size();i++)
            {
                v[i]^=e;
            }
        }
    }
    return v;
}

//a2 to do xor with suffix val we do xor with prefix and all 
vector<int> xorQuery(vector<vector<int>> &queries)
{
    // Create an empty array ans
    vector<int>ans;

    // Create a variable flag
    int flag = 0;

    // Iterate over the queries
    // If the query is of type 1 then insert at the back of the array ans (queries[i][1] ^ Val)
    // Otherwise, update the value of the flag as flag ^ queries[i][1]
    for (int i = 0; i < queries.size(); i++)
    {
        if (queries[i][0] == 1)
        {
            ans.push_back(queries[i][1]^flag);
        }
        else
        {
            flag ^= queries[i][1];
        }

    }

    // Iterate through the array ans and for each element in the array update it as ans[i] = ans[i] ^ flag
    for (int i = 0; i < ans.size(); i++)
    {
        ans[i] = ans[i] ^ flag;
    }

    // Return the array ans
    return ans;
}

//a3 we want to xor with values after we added the num
    //so storing all values coming after num in flag by doing flag^=val
#include <bits/stdc++.h> 
vector<int> xorQuery(vector<vector<int>> &queries)
{
    // Write your code here
    vector<int> v;
    int f=0;
    for(int i=queries.size()-1;i>=0;i--)//last to first
    {
        if(queries[i][0]==1)
        {
            v.push_back(queries[i][1]^f);
        }
        else 
        {
            f^=queries[i][1];
        }
    }
    reverse(v.begin(),v.end());
    return v;
}

//16 sum of infinite array
/*    Time Complexity:O(Q*(R-L))    Space Complexity:O(1).        Where Q is the number of given queries, and L and R are the given two indexes in each query. */

int mod = 1000000007;

vector<int> sumInRanges(vector<int> &arr, int n, vector<vector<long long>> &queries, int q) {

   // It stores answer for each query.    
    vector<int> ans;

   // Traversing the given queries.    

for (int i = 0; i < queries.size(); i++) {               

 vector<long long> range = queries[i];        

 

long long l = range[0] - 1;        

long long r = range[1] - 1;

       // It stores the sum       

 long long sum = 0;

       for (long long i = l; i <= r; i++) {            

                              int index = (int) (i % n);            

                              sum = (sum + arr[index]) % mod;

        }        

        sum %= mod; 

       // Add answer to each query 

       ans.push_back((int) sum);

    }

    return ans;    

 }

 //a2
 /*
    Time Complexity:O(Q+N).
    Space Complexity:O(N).

    Where N is the size of the given array, and Q is the number of queries given.
*/

int mod = 1000000007;

// Function to calculate prefix sum upto index x of the infite array.
long long func(vector<long long> &sumArray, long long x, long long n) {

    // Number of times the given array comes completely upto index x.
    long long count = (x / n) % mod;

    long long res = (count * sumArray[(int) n]) % mod;

    // Adding the remaining elements sum.
    res = (res + sumArray[(int) (x % n)]) % mod;

    return res;
}

vector<int> sumInRanges(vector<int> &arr, int n, vector<vector<long long>> &queries, int q) {

    // It stores answer for each query.
    vector<int> ans;

    // It store cumulative sum where sumArray[i] = sum(A[0]+..A[i]).
    vector<long long> sumArray(n + 1);

    for (int i = 1; i <= n; i++) {
        sumArray[i] = (sumArray[i - 1] + arr[i - 1]) % mod;
    }

    // Traversing the given queries.
    for (int i = 0; i < queries.size(); i++) {
        vector<long long> range = queries[i];
        long long l = range[0];
        long long r = range[1];

        // It stores the prefix sum from index 0 to index r in an infinite array.
        long long rsum = func(sumArray, r, n);

        // It stores the prefix sum from index 0 to index l-1 in an infinite array.
        long long lsum = func(sumArray, l - 1, n);

        // Add answer for each query.
        ans.push_back((int) ((rsum - lsum + mod) % mod));

    }

    return ans;
    
}

//17 product of array except self
#define ll long long
int mod=1e9+7;
int *getProductArrayExceptSelf(int *arr, int n)
{
    //Write your code here
    if(n==0)
    {
        int *a=new int[n];
        return a;
    }
    if(n==1)
    {
        int *a=new int[n];
        a[0]=1;
        return a;
    }
    ll l[n],r[n];
    int* a=new int[n];
    l[0]=r[n-1]=1;
    for(int i=1;i<n;i++)
    {
        l[i]=(l[i-1]*arr[i-1])%mod;
    }
    for(int i=n-2;i>=0;i--)
    {
        r[i]=(r[i+1]*arr[i+1])%mod;
    }
    for(int i=0;i<n;i++)
    {
        a[i]=(l[i]*r[i])%mod;
    }
    return a;
    
}

//18 array initialisation 

// int arr[5];//garbage
// int arr[5]={};//all 0
//int arr[5]={2,3};//2 3 0 0 0
//int arr[5]={};
//  fill_n(arr,3,10);//10 10 10 0 0 

//19 count all subarray whose sum is divisible by k
#include <bits/stdc++.h> 
int subArrayCount(vector<int> &arr, int k) {
    // Write your code here.
    int n=arr.size();
    if(n==0) return 0;
    vector<int> f(k);//frequency array . we can use hashmap as well
    f[0]=1;//empty subarray 0 sum so 0 remainder
    int c=0,rem;
    long long s=0;
    for(int i=0;i<n;i++)
    {
        s+=arr[i];
        rem=((s%k) + k)%k;
        c+=f[rem]++;
    }
    return c;
}

//20 Pair sum
#include <bits/stdc++.h> 
vector<vector<int>> pairSum(vector<int> &arr, int s){
       // Write your code here.
    sort(arr.begin(), arr.end());
    vector<vector<int>> ans ;
    //int c=0;
    for(int i=0;i<arr.size()-1;i++)
    {
        for(int j=i+1;j<arr.size();j++)
        {
            if(arr[i]+arr[j]==s) 
            {
                vector<int> n{arr[i],arr[j]};
                ans.push_back(n);
            }
            else if(arr[i]+arr[j]>s) break;
        }
    }
    return ans;
}

//better
#include <bits/stdc++.h> 
vector<vector<int>> pairSum(vector<int> &arr, int s){
   // Write your code here.
    unordered_map<int,int> m;
    sort(arr.begin(),arr.end(),greater<int>());
    vector<vector<int>> ans;
    int n=arr.size();

    int c=0;
    int p=0;
    for(int i=0;i<n;i++)
    {
        p=s-arr[i];//partner 
        if(m[p])
        {
            c=m[p];
            while(c--)
            {
                vector<int> v{arr[i],p};
                ans.push_back(v);
            }

        }
        m[arr[i]]++;
    }
    sort(ans.begin(),ans.end());
    return ans;
}

//21 Valid pairs -> return true if array can be divided into pairs whose when divided by k is m
#include <bits/stdc++.h> 
bool isValidPair(vector<int> &arr, int n, int k, int m)
{
    // Write your code here.
    if(n&1) return false;
    int r;
    unordered_map<int,int> mp;
    for(int i=0;i<n;i++)
    {
        r=(arr[i]%k);
        mp[r]++;
    }

    // int c1,c2;
    // for(int i=0;i<n;i++)
    // {
    //     c1=mp[arr[i]%k];
    //     if(arr[i]%k<=m)
    //     {
    //         c2=mp[m-(arr[i]%k)];
    //     }
    //     else
    //     {
    //         c2=mp[k-(arr[i]%k)+m];
    //     }
    //     if(c1!=c2) return false;
    // }
    // return true;

    int c1,c2,r1;
    for(auto ma:mp)
    {
        r1=ma.first;
        c1=ma.second;
        c2=mp[(m-r1+k)%k];
        if(c1!=c2) return false;
        mp.erase((m-r1+k)%k);
    }
    return true;
}

/*
    Time Complexity: O(N)
    Space Complexity: O(N)
    
    Where 'N' denotes the length of the array.
*/
//a2
#include <unordered_map>

bool isValidPair(vector<int> &arr, int n, int k, int m) 
{
    
    // An odd length array cannot be divided into pairs.
    if (n % 2 == 1) 
    {
        return false;
    }

    /*
        Create a frequency array to count occurrences
        of all remainders when divided by k.
    */
    unordered_map<int, int> map;

    for (int i = 0; i < n; i++) 
    {
        int rem = arr[i] % k;
        map[rem]++;
    }

    unordered_map<int, int>:: iterator itr;

    for(itr = map.begin(); itr != map.end(); itr++)  
    {
        int rem = itr->first;

        /*
            If current remainder divides
            m into two halves.
        */
        if (2 * rem == m) 
        {

            // Then there must be even occurrences of such remainder.
            if (map[rem] % 2 == 1)
            {
                return false;
            }
        }

        /* 
            Else number of occurrences of remainder
            must be equal to number of occurrences of m - remainder.
        */
        else 
        {
            if (map[(m - rem + k) % k] != map[rem]) 
            {
                return false;
            }
        }
    }

    return true;
}

//22 Max product count quadruples a*b=c*d
#include <bits/stdc++.h> 
vector<long long> maxProductCount(vector<int> &arr, int n) {
    // Write your code here.
    map<long long,long long> mp;
    long long p;
    for(int i=0;i<n;i++)
    {
        for(int j=i+1;j<n;j++)
        {
            p=1ll*arr[i]*arr[j];
            mp[p]++;
        }
    }

    long long a1=0,a2=1;
    
    for(auto em:mp)
    {
        if(em.second>a2)
        {
            a1=em.first;
            a2=em.second;
        }
    }
    //cout<<endl;
    vector<long long> ans;
    if(a1==0) 
    {
        ans.push_back(0);
        return ans;
    }
    a2=((a2)*(a2-1))/2;
    ans.push_back(a1);
    ans.push_back(a2);
    return ans;
}

//23 Buy and sell stock infinite transaction one after other
long getMaximumProfit(long *values, int n)
{
    //Write your code here
    int bd=0,sd=0;
    long p=0;
    for(int i=1;i<n;i++)
    {
        if(values[i]<=values[sd] )
        {
            p+=values[sd]-values[bd];
            bd=i;
            sd=i;
        }
        else if(values[i]>values[sd])
        {
            sd=i;
        }
    }
    p+=values[sd]-values[bd];
    return p;
}
//a2
/*
    Time Complexity : O(N)
    Space Complexity : O(1)
    
    where N is the total number of days.
*/

long getMaximumProfit(long *values, int n)
{
    // If the data is only for one day, we simply return 0 because we can't sell if we buy on day 0
    if (n <= 1)
    {
        return 0;
    }

    long profit = 0L;
    int buyingDay = 0, sellingDay = 1;
    int totalDays = n;

    while (sellingDay < totalDays)
    {
        // If the value of the stock is greater than the buying day, sell the stock
        if (values[sellingDay] > values[buyingDay])
        {
            // Add the profit earned by selling the stock.
            profit += (values[sellingDay] - values[buyingDay]);
        }
        buyingDay++;
        sellingDay++;
    }

    return profit;
}

//24 Non decreasing array
#include <bits/stdc++.h> 
bool isPossible(int *arr, int n)
{
    //  Write your code here.
    int pos=-1;
    for(int i=0;i<n-1;i++)
    {
        if(arr[i]>arr[i+1])
        {
            if(pos!=-1) return false;
            pos=i;
        }
    }

    return pos==-1 || pos==0 || pos==(n-2) || arr[pos-1]<=arr[pos+1] || arr[pos]<=arr[pos+2] ;
}

//25 Longest subsequence 
//2 3 5 3 1 6 
//sort 1 2 3 3 5 6 
//if diff 1 c++ if diff 0 same c else c=1

//tc O(nlogn) sc O(1)
#include <bits/stdc++.h> 
int lengthOfLongestConsecutiveSequence(vector<int> &arr, int n) {
    // Write your code here.
    sort(arr.begin(),arr.end());
    int c=1,ans=1;
    for(int i=1;i<n;i++)
    {
        if(arr[i]==arr[i-1]+1) 
        {
            c++;
            ans=max(ans,c);
        }
        else if(arr[i]!=arr[i-1]) c=1;
    }
    return ans;
}

//a2 tc O(n) sc O(n)
/*  
    Time Complexity: O(N)
    Space Complexity: O(N)

    Where N is the length of the given array.
*/

#include <unordered_set>

int lengthOfLongestConsecutiveSequence(vector<int> &arr, int n) {
    // To store length of longest consecutive sequence.
    int mx = 0;

    // To store the length of current consecutive Sequence.
    int count = 0;

    // To store all the unique elements of array.
    unordered_set<int> set;

    for (int i = 0; i < n; i++) {
        set.insert(arr[i]);
    }

    for (int i = 0; i < n; i++) {
        int previousConsecutiveElement = arr[i] - 1;

        if (set.find(previousConsecutiveElement) == set.end()) {

            // 'arr[i]' is the first value of consecutive sequence.
            int j = arr[i];
            
            while (set.find(j) != set.end()) {
                // The next consecutive element by will be j + 1.
                j++;
            }

            // Update maximum length of consecutive sequence.
            mx = max(mx, j - arr[i]);
        }

    }

    return mx;
}

//26 Print array after k operations

#include <bits/stdc++.h> 
vector<int> printArrayAfterKOperations(vector<int> &Arr, int N, int K) {

    // Write your code here.
    if(K==0) return Arr;
    int h=Arr[0],l=Arr[0];
    for(int i=1;i<N;i++)
    {
        if(Arr[i]>h) h=Arr[i];
        else if(Arr[i]<l) l=Arr[i];
    }
    int t=2;
    while(t--)
    {
        for(int i=0;i<N;i++)
        {
            Arr[i]=h-Arr[i];//(h-l)-(h-Arr[i])= Arr[i]-l
        }
        if(K&1) return Arr;
        h-=l;   
    }
    
    return Arr;
}

//a2 
#include <bits/stdc++.h> 
vector<int> printArrayAfterKOperations(vector<int> &Arr, int N, int K) {

    // Write your code here.
    if(K==0) return Arr;

    if(K&1)
    {
        int h=Arr[0];
        for(int i=1;i<N;i++)
        {
            if(Arr[i]>h) h=Arr[i];
        }

        for(int i=0;i<N;i++)
        {
            Arr[i]=h-Arr[i];
        }
    }
    else
    {
        int l=Arr[0];
        for(int i=1;i<N;i++)
        {
            if(Arr[i]<l) l=Arr[i];
        }

        for(int i=0;i<N;i++)
        {
            Arr[i]-=l;
        }

    }
    return Arr;
    
}

//27 Minimum number of platforms
int calculateMinPatforms(int at[], int dt[], int n) {
    // Write your code here.
    sort(at,at+n);
    sort(dt,dt+n);
    int ans=1;//1st train arrived 
    int a=1,d=0;//check between 2nd arrival and 1st depart
    while(a<n)
    {
        if(at[a]<=dt[d]) ans++;//new plat
        else d++;//one depart no need of new
        a++;
    }
    return ans;
}

//28 element that occurs more than n/3 times
 
#include <bits/stdc++.h> 
vector<int> majorityElementII(vector<int> &arr)
{
    // Write your code here.
    int n=arr.size();
    vector<int> ans;
    unordered_map<int,int> um;
    for(int val:arr)
    {
        um[val]++;
    }
    
    for(auto x:um)
    {
        if(x.second>n/3) ans.push_back(x.first);
    }
    return ans;

}

//a2 Moore's Voting Algorithm Approach
    //if we remove 3 distinct elements from the array, the elements which occurred 
    //more than N/3 times do not change
/*
    Time Complexity: O(N)
    Space Complexity: O(1)

    Where 'N' is the number of elements in the given array/list
*/

vector<int> majorityElementII(vector<int> &arr)
{
    int n = arr.size();

    // Array for storing final answer.
    vector<int> majorityElement;

    // Variables for storing the elements which may occur more than n/3 times.
    int firstCandidate = 0, secondCandidate = 0;

    // Variables for storing the frequency of the candidate elements.
    int firstCount = 0, secondCount = 0;

    // Iterate through the array.
    for (int i = 0; i < n; i++)
    {
        // Increment firstCount if the current element is equal to firstCandidate.
        if (arr[i] == firstCandidate)
        {
            firstCount = firstCount + 1;
        }

        // Increment secondCount if the current element is equal to secondCandidate.
        else if (arr[i] == secondCandidate)
        {
            secondCount = secondCount + 1;
        }
        // Change value of the firstCandidate to the current element if firstCount is equal to 0.
        else if (firstCount == 0)
        {
            firstCandidate = arr[i];
            firstCount = 1;
        }

        // Change value of the secondCandidate to the current element if secondCount is equal to 0.
        else if (secondCount == 0)
        {
            secondCandidate = arr[i];
            secondCount = 1;
        }

        // Otherwise decrement firstCount and secondCount by 1.
        else
        {
            firstCount = firstCount - 1;
            secondCount = secondCount - 1;
        }
    }

    firstCount = 0;
    secondCount = 0;

    // Iterate through the array to find frequency of firstCandidate and secondCandidate.
    for (int i = 0; i < n; i++)
    {
        // Increment firstCount if the current element is equal to firstCandidate.
        if (arr[i] == firstCandidate)
        {
            firstCount = firstCount + 1;
        }

        // Increment secondCount if the current element is equal to secondCandidate.
        else if (arr[i] == secondCandidate)
        {
            secondCount = secondCount + 1;
        }
    }

    // Include firstCandidate in the answer if its frequency is more than n/3.
    if (firstCount > n / 3)
    {
        majorityElement.push_back(firstCandidate);
    }

    // Include secondCandidate in the answer if its frequency is more than n/3.
    if (secondCount > n / 3)
    {
        majorityElement.push_back(secondCandidate);
    }

    // Return all stored majority elements.
    return majorityElement;
}

//29 Strings
//convert lower to upper initial letter of every word
//ip - hello This is kam
//op - Hello This Is Kam
#include <bits/stdc++.h> 
string convertString(string str) 
{
    // WRITE YOUR CODE HERE
    if(str[0]>='a' && str[0]<='z')
    {
        str[0]=str[0]-'a'+'A';
    }
    for(int i=1;i<str.size();i++)
    {
        if(str[i-1]==' ' && str[i]>='a' && str[i]<='z')
        {
            str[i]=str[i]-'a'+'A';
        }
    }
    return str;
}

//30 encode
//ip aabbbc op a2b3c1
#include <bits/stdc++.h> 
string encode(string &message)
{
    //   Write your code here.
    int c=1;
    char ch=message[0];
    string a;
    for(int i=1;i<message.size();i++)
    {
        if(message[i]==ch) c++;
        else 
        {
            a.push_back(ch);
            //a+=to_string(c);
            a.append(to_string(c));
            ch=message[i];
            c=1;
        }
    }
    
    a.push_back(ch);
    a+=to_string(c);
    return a;
}

//31 Remove Vowels
// on on
#include <bits/stdc++.h> 
string removeVowels(string inputString) {
    // Write your code here.
    string ans;
    for(int i=0;i<inputString.size();i++)
    {
        char c=inputString[i];
        if(c=='a' || c=='e' || c=='i' || c=='o' || c=='u' || c=='A' || c=='E' || c=='I' || c=='O' || c=='U')
        {
            
        }
        else 
        {
            ans.push_back(c);
        }
    }
    return ans;
}

//a2 o(n) o(1)
string remVowel(string str)
{
    vector<char> vowels = {'a', 'e', 'i', 'o', 'u',
                           'A', 'E', 'I', 'O', 'U'};
     
    for (int i = 0; i < str.length(); i++)
    {
        if (find(vowels.begin(), vowels.end(),
                      str[i]) != vowels.end())
        {
            str = str.replace(i, 1, "");
            i -= 1;
        }
    }
    return str;
}

//32 Minimum number of parentheses
#include <bits/stdc++.h> 
int minimumParentheses(string pattern) {
    // Write your code here.
    // pattern is the given string.
    int o=0,c=0,ans;
    for(int i=0;i<pattern.size();i++)
    {
        char ch=pattern[i];
        if(ch=='(')
        {
            o++;
        }
        else 
        {
            if(o>0) o--;
            else ans++;
        }
    }
    ans+=o;
    return ans;
}

//33 Left & right rotation of string 
#include <bits/stdc++.h> 

void revf(string &s,int i,int j)
{
    while(i<j)
    {
        char temp=s[i];
        s[i]=s[j];
        s[j]=temp;
        i++;
        j--;
    }
}

string leftRotate(string str, int d) {
    // Write your code here.
    int n=str.size();
    d=(d+n)%n;
    if(d==0) return str;
    revf(str,0,d-1);
    revf(str,d,n-1);
    revf(str,0,n-1);
    return str;
}

string rightRotate(string str, int d) {
    // Write your code here.
    int n=str.size();
    d=(d+n)%n;
    str=leftRotate(str,n-d);
    return str;
}

//34 Length of longest substring with atmost k distinct char
#include <bits/stdc++.h> 
int getLengthofLongestSubstring(string s, int k) {
    // Write your code here.
    int j=0,ans=0;
    unordered_map<char,int> um;
    for(int i=0;i<s.size();i++)
    {
        char ch=s[i];
        um[ch]++;
        
        while(um.size()>k)
        {
            char chj=s[j];
            um[chj]--;
            if(um[chj]==0) um.erase(chj);
            j++;
        }
        
        int len=i-j+1;
        ans=max(ans,len);
    }
    return ans;
}

//35 Min steps to become anagram
#include <bits/stdc++.h> 
int getMinimumAnagramDifference(string &str1, string &str2) {
    // Write your code here.
    vector<int> v(26,0);
    int n=str1.size();
    int ans=0;
    for(int i=0;i<n;i++)
    {
        v[str1[i]-'a']++;
    }
    for(int i=0;i<n;i++)
    {
        if(v[str2[i]-'a']==0) ans++;
        else v[str2[i]-'a']--;
    }
    return ans;
}

//a2 using map
#include <bits/stdc++.h> 
int getMinimumAnagramDifference(string &str1, string &str2) {
    // Write your code here.
    unordered_map<char,int> um;
    int n=str1.size();
    int ans=0;
    for(int i=0;i<n;i++)
    {
        um[str1[i]]++;
    }
    for(int i=0;i<n;i++)
    {
        if(um[str2[i]]==0) ans++;
        else um[str2[i]] --;
    }
    return ans;
}

//36 min no of preprocessing 
/*
    Time complexity: O(N)
    Space complexity: O(1)

    Where 'N' is the length of the string.
*/

int minimumOperations(string &a, string &b){

    if (a.size() != b.size())
    {
        return -1;
    }

    // Length of the given string.
    int n = a.size();

    // To store the required answer.
    int count = 0;

    char c1, c2, c3, c4;

    // Run a loop upto 'n'/2.
    for (int i = 0; i < n / 2; i++)
    {

        // Collect the group.
        c1 = a[i];
        c2 = a[n - i - 1];
        c3 = b[i];
        c4 = b[n - i - 1];

        // Cases that doesn't require any preprocessing move.
        if ((c1 == c2 && c3 == c4) || (c1 == c3 && c2 == c4) || (c1 == c4 && c2 == c3))
        {
            continue;
        }

        // Cases that require only one preprocessing move.
        else if (c1 == c3 || c1 == c4 || c2 == c3 || c2 == c4 || c3 == c4)
        {
            count++;
        }

        // All remaining cases require 2 changes.
        else
        {
            count += 2;
        }
    }

    // If 'n' is odd.
    if (n % 2 == 1 && a[n / 2] != b[n / 2])
    {
        count++;
    }

    return count;
}

//37 word pattern match
#include <bits/stdc++.h> 

bool check(string w,string p)
{
    if(w.size()!=p.size()) return false;
    unordered_map<char,char> um;//pattern char vs word char 
    for(int i=0;i<p.size();i++)
    {
        if(um.find(p[i])==um.end())
        {
            um[p[i]]=w[i];
        }
        else 
        {
            if(um[p[i]]!=w[i]) return false;
        }
    }
    //now check if multiple char of pattern are not mapped to same char of word
    //eg word ccc pattern foo ->c mapped to f & o
    unordered_set<char> us;
    for(auto x:um)
    {
        char ch=x.second;
        if(us.find(ch)!=us.end())
        {
            return false;
        }
        else 
        {
            us.insert(ch);
        }
    }
    return true;
}

vector<string> matchSpecificPattern(vector<string> words, int n, string pattern)
{
    // Write your code here.
    vector<string> ans;
    for(int i=0;i<words.size();i++)
    {
        if(check(words[i],pattern))
        {
            ans.push_back(words[i]);
        }
    }
    return ans ;
}

//38 extra question 
//given a palindrome no -> find next palindrome greater than given no
#include <bits/stdc++.h> 
string nextLargestPalindrome(string s, int length){
    // Write your code here.
    int h=length/2;
    int n=length;
    while(h<length)
    {
        if(s[h]<'9')
        {
            
            int idx=h;
            s[idx]++;
            if(idx != n-idx-1) s[n-idx-1]++;
            
            idx--;
            
            while(idx>=(length/2))
            {
                s[idx]='0';
                s[n-1-idx]='0';
                idx--;
            }
            break;
        }
        h++;
    }
    if(h>=length)
    {
        s[0]='1';
        for(int i=1;i<length;i++) s[i]='0';
        s.push_back('1');
    }
    return s;
}

//39 Next Palindrome
#include <bits/stdc++.h> 
string mirror(string str)
{
    int i=0,j=str.size()-1;
    while(i<j)
    {
        str[j--]=str[i++];
    }
    return str;
}

string nextLargestPalindrome(string s, int length){
    // Write your code here.
    string ans=mirror(s);
    if(ans>s) return ans;

    int midx=(length-1)/2;

    for(int i=midx;i>=0;i--)
    {
        if(ans[i]!='9')
        {
            ans[i]++;
            break;
        }
        else 
        {
            ans[i]='0';
        }
    }

    ans=mirror(ans);

    if(ans[0]=='0')
    {
        ans[0]='1';
        ans+='1';
    }

    return ans;

}

//a2 all test case passed
#include<bits/stdc++.h>

string helperForEvenLength(string &s,int i,int j){
    while(i>=0 && j<s.size()){
        if(s[i]!='9'&& s[j]!='9'){
            s[i]  = ((s[i]-48)+1)+48;
            
            s[j] = ((s[j]-48)+1)+48;
            
            break;
        }else{
            s[i] = '0';
            s[j] = '0';
            i--;
            j++;
        }
    }
    return s;
}
bool isPalindrome(string &s,int l){
    int i = 0;
    int j = l-1;
    while(i<j){
        if(s[i]!=s[j]){
            return false;
        }
        i++;
        j--;
    }
    return true;
}
void makePalindrome(string &s,int l){
    int i = 0;
    int j = l-1;
    while(i<j){
        s[j] = s[i];
        i++;
        j--;
    }
}
bool biggerThenOriginal(string &s,string &temp,int length){
    for(int i=0;i<length;i++){
        if(s[i]<temp[i]){
            return false;
        }else if(s[i]>temp[i]){
            return true;
        }
    }
    return true;
}
string nextLargestPalindrome(string s, int length){
    // Write your code here.
    string temp = s;
    bool check = isPalindrome(s,length);
    if(check == false){
        makePalindrome(s,length);
        bool isBig = biggerThenOriginal(s,temp,length);
    
        if(isBig==true){
           return s;
        }
    }
    
    bool allnine = true;
    for(int i=0;i<length;i++){
        if(s[i]!='9'){
            allnine = false;
            break;
        }
    }
    string ans = "";
    if(allnine){
        for(int i=0;i<length-1;i++){
            ans+='0';
        }
        ans = '1'+ans+'1';
        return ans;
    }else if(length%2==0){
        int i = length/2-1;
        int j = length/2;
        return helperForEvenLength(s,i,j);
    }else{
        int mid = length/2;
        int i = mid-1;
        int j = mid+1;
        if(s[mid]!='9'){
            s[mid] = ((s[mid]-48)+1)+48;
            return s;
        }else{
            s[mid] = '0';
            return helperForEvenLength(s,i,j);
        }
    }
    
}

//a3 
//999 9999 
//129921 12921 12321 123321
//783322 713322 94187978322

#include <bits/stdc++.h> 

bool checkPal(string &s)
{
    int i=0,j=s.size()-1;
    while(i<j)
    {
        if(s[i++]!=s[j--]) return false;
    }
    return true;
}

bool check9(string &s)
{
    for(auto x:s)
    {
        if(x!='9') return false;
    }
    return true;
}

void add1(string &s)
{
    int i=(s.size()-1)/2;
    //int j=i;
    while(i>=0)
    {
        if(s[i]<'9')
        {
            s[i]++;
            break;
        }
        else
        {
            s[i]='0';
        }
        i--;
    }
    // i++;
    // while(i<=j)
    // {
    //  s[i]='0';
    //  i++;
    // }
}

void mirror(string &s)
{
    int i=0,j=s.size()-1;
    while(i<j)
    {
        s[j--]=s[i++];
    }
}

string nextLargestPalindrome(string s, int l){
    // Write your code here.
    bool isPal=checkPal(s);
    if(isPal)
    {
        bool all9=check9(s);
        if(all9)
        {
            s[0]='1';
            for(int i=1;i<s.size();i++)
            {
                s[i]='0';
            }
            s.push_back('1');
            return s;
        }
        else
        {
            add1(s);
            mirror(s);
            return s;
        }
    }
    string cp=s;
    mirror(s);
    if(s>cp) return s;
    add1(s);
    mirror(s);
    return s;
}

//40 First Unique Char
#include <bits/stdc++.h> 
char findNonRepeating(string str) {
    // Write your code here.
    unordered_map<char,int> um;
    for(int i=0;i<str.size();i++)
    {
        char ch=str[i];
        um[ch]++;
    }
    for(int i=0;i<str.size();i++)
    {
        char ch=str[i];
        if(um[ch]==1) return ch;
    }
    return '#';
}

//a2 
#include <bits/stdc++.h> 
char findNonRepeating(string str) {
    // Write your code here.
    vector<int> v(26);
    for(int i=0;i<str.size();i++)
    {
        char ch=str[i];
        v[ch-'a']++;
    }
    for(int i=0;i<str.size();i++)
    {
        char ch=str[i];
        if(v[ch-'a']==1) return ch;
    }
    return '#';
}

//41 Compare Versions
// 1 if first greater ,-1 if first smaller , 0 if equal
// 123.45
// 123      1
// 1.0.0
// 1        0

#include <bits/stdc++.h> 
#define ll long long
int compareVersions(string a, string b) 
{
    // Write your code here
    int i=0,j=0;
    while(i<a.size() || j<b.size())
    {
        ll a1=0,b1=0;
        while(i<a.size() && a[i]!='.')
        {
            int ch=a[i]-'0';
            a1=a1*10+ch;
            i++;
        }
        while(j<b.size() && b[j]!='.')
        {
            int ch=b[j]-'0';
            b1=b1*10+ch;
            j++;
        }
        if(a1>b1) return 1;
        else if(a1<b1) return -1;
        i++;
        j++;
    }
    return 0;
}

//42 kth char in decrypted string
//a2b3cd3
//8 c
//9 d 
//12 $
#define ll long long
char kThCharaterOfDecryptedString(string s, long long k)
{
    //  Write your code here.
    string s2="";
    ll len=0,cs=0;
    for(int i=0;i<s.size();)
    {
        while(i<s.size() && s[i]>='a' && s[i]<='z')
        {
            s2.push_back(s[i]);
            i++;
        }
        while(i<s.size() && s[i]>='0' && s[i]<='9')
        {
            int val=s[i]-'0';
            cs=cs*10+val;
            i++;
        }
        if((len+cs*s2.size()) < k) 
        {
            len+=cs*s2.size();
            s2.clear();
            cs=0;
        }
        else if((len+cs*s2.size()) == k) return s2.back();
        else 
        {
            ll left=k-len;
            if(left%s2.size() == 0) return s2.back();
            ll rem=left%s2.size();
            return s2[rem-1];
        }
    }
    return '$';
}

//43 multiply two strings
#include<bits/stdc++.h>
string multiplyStrings(string a , string b ){
    //Write your code here
    if(a=="0" || b=="0") return "0";
    if(a=="1") return b;
    if(b=="1") return a;
    int n1=a.size(),n2=b.size();
    vector<int> v(n1+n2);

    for(int i=n1-1;i>=0;i--)
    {
        for(int j=n2-1;j>=0;j--)
        {
            v[i+j+1]+=(a[i]-'0')*(b[j]-'0');
            v[i+j]+=v[i+j+1]/10;
            v[i+j+1]%=10;
        }
    }

    string ans="";
    int idx=0;
    while(idx<v.size() && v[idx]==0) idx++;
    for(int i=idx;i<v.size();i++) ans.push_back(v[i]+'0');
    return ans;
}

//a2
    for(int i=0;i<n1;i++)
    {
        for(int j=0;j<n2;j++)
        {
            v[i+j+1]+=(a[i]-'0')*(b[j]-'0');
            // v[i+j]+=v[i+j+1]/10;
            // v[i+j+1]%=10;
        }
    }
    int n=v.size();

    for(int i=n-2;i>=0;i--)
    {
        v[i]+=v[i+1]/10;
        v[i+1]%=10;
    }
/*
  234
   56
-----
01234 idx 
   24
  18
 12
  20
 15
10
-----
13104   
*/

//44 valid ipv4 address
#include <bits/stdc++.h> 
bool isValidIPv4(string ipAddress) {
    // Write your code here.
    int p=0,n=ipAddress.size();
    for(int i=0;i<n;)
    {
        //char ch=ipAddress[i];
        bool flag=0;
        int val=0;
        while(i<n && ipAddress[i]>='0' && ipAddress[i]<='9')
        {
            flag=1;
            
            val=val*10 + (ipAddress[i]-'0');
            i++;
        }
        //cout<<val<<" ";
        if(flag) p++;
        if(p>4 || val<0 || val>255) return false;
        if(i<n && ipAddress[i]!='.') return false;
        i++;
        
    }
    if(p==4) return true;
    else return false;
}

//a2 to check 00 01 etc
/*
    Time complexity: O(1)
    Space complexity: O(1)

    We are using constant time and space. 
*/

bool isValidIPv4(string ipAddress) {
    int dots = 0;            // To store total number of dots
    int currentNumber = -1;  // Integer form of every single part

    for (int i = 0; i < ipAddress.size(); i++) {
        // If current character is '.' then check 'currentNumber' is valid or not.
        if (ipAddress[i] == '.') {
            // If 'currentNumer' is valid then its must be, from '0' to '255'
            if (currentNumber < 0 or currentNumber > 255) {
                return false;
            }

            currentNumber = -1;
            dots += 1;
        } else {
            // First check given character is valid number or not.
            if (!isdigit(ipAddress[i])) {
                return false;
            }
            if (currentNumber == -1) {
                currentNumber = ipAddress[i] - '0';
            } else {
                currentNumber = currentNumber * 10 + (ipAddress[i] - '0');
            }
            /*
                If any substring is '00' then it can't be valid.
                According to 'currentNumber' it is '0' and valid, so we need to check it extra.

            */
            if (i > 0) {
                if (ipAddress[i - 1] == '0') {
                    if (i > 1) {
                        if (!isdigit(ipAddress[i - 2]) || ipAddress[i -2] == '0') {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }
        }
    }
    // If number of dots are '3' and every part is valid only then return 'True'
    if (currentNumber < 0 or currentNumber > 255 or dots != 3) {
        return false;
    }
    return true;
}

//45 Longest Mountain Subarray
#include <bits/stdc++.h> 
int longestMountain(int *arr, int n)
{
    // Write your code here.
    if(n<3) return 0;
    int ans=0;
    int i=0,j=1;
    while(j<n)
    {
        bool f1=0,f2=0;
        while(j<n && arr[j]>arr[j-1]) 
        {
            f1=1;
            j++;
        }
        while(f1 && j<n && arr[j]<arr[j-1])
        {
            f2=1;
            j++;
        }
        if(f2)
        {
            int cl=j-i;
            ans=max(ans,cl);
//             f1=0;
//             f2=0;
        }
        i=max(j-1,i+1);
        j=i+1;
    }
    return ans;
}

//a2
int longestMountain(int *arr, int n)
{
    // Write your code here.
    if(n<3) return 0;
    int ans=0;
    int i=0,j=1;
    while(j<n)
    {
        while(i+1<n && arr[i]==arr[i+1]) i++;
        j=i+1;
        bool f1=0,f2=0;
        while(j<n && arr[j]>arr[j-1]) 
        {
            f1=1;
            j++;
        }
        while(j<n && arr[j]<arr[j-1])
        {
            f2=1;
            j++;
        }
        if(f1 && f2)
        {
            int cl=j-i;
            ans=max(ans,cl);
//             f1=0;
//             f2=0;
        }
        i=max(j-1,i+1);
        j=i+1;
    }
    return ans;
}

//46 Triplets with zero sum
#include <bits/stdc++.h> 
vector<vector<int>> findTriplets(vector<int>arr, int n) {
    // Write your code here
    sort(arr.begin(),arr.end());
    vector<vector<int>> ans;
    for(int i=0;i<n-2;i++)
    {
        int tar=-arr[i];
        int j=i+1,k=n-1;
        while(j<k)
        {
            int s=arr[j]+arr[k];
            if(s>tar) k--;
            else if(s<tar) j++;
            else 
            {
                ans.push_back({arr[i],arr[j],arr[k]});
                int front=arr[j],back=arr[k];
                while(j<k && arr[j]==front) j++; 
                while(j<k && arr[k]==back) k--; 
            }
            
        }
        while(i<n-2 && arr[i+1]==arr[i]) i++;
    }
    return ans;
}

//47 Container with most water(maxarea)
int maxArea(vector<int>& height) {
    // Write your code here.
    if(height.size()==0 || height.size()==1) return 0;
    
    int i=0,j=height.size()-1;
    int ans=0;
    while(i<j)
    {
        //int minh
        int ca=(j-i)*min(height[i],height[j]);
        ans=max(ans,ca);
        if(height[i]>height[j]) j--;
        else i++;
    }
    return ans;
}

//48 Sum of two elements equal to third
#include <bits/stdc++.h> 
vector<int> findTriplets(vector<int> &arr, int n) 
{
    //Write your code here.
    //vector<int> v;
    sort(arr.begin(),arr.end());
    for(int k=n-1;k>=2;k--)
    {
        int i=0,j=k-1;
        while(i<j)
        {
            int su=arr[i]+arr[j];
            if(su==arr[k])
            {
                // v.push_back(arr[i]);
                // v.push_back(arr[j]);
                // v.push_back(arr[k]);
                // return v;
                return {arr[i],arr[j],arr[k]};
            }
            else if(su<arr[k]) i++;
            else j--;
        }
        while(k>=2 && arr[k-1]==arr[k]) k--;
        
    }
    //return v;
    return {};
}

//49 3Sum
#include <bits/stdc++.h> 
vector<vector<int>> findTriplets(vector<int>arr, int n, int K) {
    // Write your code here.
    sort(arr.begin(),arr.end());
    vector<vector<int> > v;
    for(int i=0;i<n-2;i++)
    {
        int tar=K-arr[i];
        int j=i+1,k=n-1;
        while(j<k)
        {
            int s=arr[j]+arr[k];
            if(s>tar) k--;
            else if(s<tar) j++;
            else
            {
                v.push_back({arr[i],arr[j],arr[k]});
                int fr=arr[j],ba=arr[k];
                j++;
                k--;
                while(j<k && arr[j]==fr) j++;
                while(j<k && arr[k]==ba) k--;
            }
        }
        while(i<n-1 && arr[i+1]==arr[i]) i++;
    }
    return v;
}

//50 Shortest substring with all char 
//if multiple shortest length select substring starting first
#include <bits/stdc++.h> 
string shortestSubstring(string s)
{
    // Write your code here.
    unordered_set<char> us;
    for(char ch:s) us.insert(ch);
    int sz=us.size();
    //cout<<sz<<endl;
    if(sz==s.size()) 
    {
        //cout<<"v"<<endl;
        return s;
    }
    
    int n=s.size();
    int e=0;
    string ans=s;
    //cout<<"K"<<ans<<" ";
    unordered_map<char,int> um;
    for(int i=0;i<=n-sz;i++)
    {
        while(e<n && um.size()<sz)
        {
            char ch=s[e];
            um[ch]++;
            e++;
        }
        int len=e-i;
        if(um.size()==sz && len<ans.size()) 
        {
            //cout<<len<<" ";
            ans=s.substr(i,e-i);
        }
        
        if(um[s[i]]==1) um.erase(s[i]);
        else um[s[i]]--;
    }
    return ans;
}

//51 Valid Parenthesis ( ) *
#include <bits/stdc++.h> 
bool checkValidString(string &s){
    // Write your code here.
    stack<int> oi,si;//stack to hold index of opening and stars
    for(int i=0;i<s.size();i++)
    {
        char ch=s[i];
        if(ch=='(') oi.push(i);
        else if(ch=='*') si.push(i);
        else 
        {
            if(oi.size()>0) oi.pop();
            else if(si.size()>0) si.pop();
            else return false;
        }
    }
    while(oi.size()>0)
    {
        if(si.size()==0) return false;
        if(si.top()<oi.top()) return false;
        oi.pop();
        si.pop();
    }
    return true;
}

//a2 
bool checkValidString(string &s){
    int n = s.length()-1;
        int opencount = 0, closecount = 0;
        for(int i=0; i<=n; i++){
            if(s[i]=='(' || s[i]=='*')
                opencount++;
            else opencount--;
            if(s[n-i]==')' || s[n-i]=='*')
                closecount++;
            else closecount--;
            if(opencount<0 || closecount<0) return false;
        }
        return true;
}

//52 remove consecutive duplicates
//tc and sc O(n) O(n)
string removeConsecutiveDuplicates(string str) 
{
    string ans;
    ans+=str[0];
    for(int i=1;i<str.size();i++)
    {
        char c=str[i];
        if(c!=ans.back()) ans+=c;
    }
    return ans;
}

//a2 O(n) O(1)
string removeConsecutiveDuplicates(string str) 
{
    //Write your code here
    int i=0,j=1,n=str.size();
    while(j<n)
    {
        if(str[i]!=str[j])
        {
            i++;
            str[i]=str[j];
        }
        j++;
    }
    return str.substr(0,i+1);
    
}

//53 minimum operation to make string identical
//operation take a char and put it to end in string1
//O(n) O(n)
#include <bits/stdc++.h>

int minCostToGivenString(string str1, string str2)
{
    // Write your Code here
    int s1=str1.size(),s2=str2.size();
    if(s1!=s2) return -1;
    
    unordered_map<char,int> um;
    for(char c:str1) um[c]++;
    
    for(char c:str2) um[c]--;
    
    for(auto x:um)
    {
        if(x.second!=0) return -1;
    }
    
    int i=0,j=0,ans=0;
    while(i<s1 && j<s2)
    {
        if(str1[i]==str2[j])
        {
            i++;
            j++;
        }
        else 
        {
            i++;
            ans++;
        }
    }
    return ans ;
}


//a2 O(n) O(1)
int minCostToGivenString(string str1, string str2)
{
    int freq[52]={0};
    for (int i = 0; i < str1.size(); i++)
    {
        if (str1[i] >= 'a' && str1[i] <= 'z')
        {
            freq[str1[i] - 'a']++;
        }
        else
        {
            freq[str1[i] - 'A' + 26]++;
        }
    }

    for (int i = 0; i < str2.size(); i++)
    {
        if (str2[i] >= 'a' && str2[i] <= 'z')
        {
            freq[str2[i] - 'a']--;
        }
        else
        {
            freq[str2[i] - 'A' + 26]--;
        }
    }

    for (int i = 0; i < 52; i++)
    {
        if (freq[i]) 
        {
            return -1;
        }
    }

    int i = 0, j = 0, ans = 0;
    // i points to str1 and j points to str2
    while (i < str1.size() && j < str2.size())
    {
        if (str1[i] == str2[j])
        {
            i++;
            j++;
        }
        else
        {
            i++;
            ans++;
        }
    }

    return ans;
}

//54 ASCII VALUES
A 65 Z 90
a 97 z 122

//55 is s1 subsequence of s2 
// ab is subsequence of acbd but not of bcad
//O(str2.size()) O(str1.size())
#include <bits/stdc++.h> 
bool isSubSequence(string str1, string str2) {
    // Write your code here.
    queue<char> q;
    for(char c:str1) q.push(c);
    
    for(char c:str2)
    {
        if(q.size()==0) return true;
        if(c==q.front()) q.pop();
    }
    
    return q.size()==0;
}

//a2 O(str2.size()) O(1)
#include <bits/stdc++.h> 
bool isSubSequence(string str1, string str2) {
    // Write your code here.
    int i=0,j=0;
    int s1=str1.size(),s2=str2.size();
    while(i<s1 && j<s2)
    {
        if(str1[i]==str2[j++]) i++;
        //j++;
    }
    return i==s1;
}

//56 smallest subarray with k distinct elements//leftmost
//O(n) O(k)
// 6 2 
// 1 2 2 3 1 3
// 0 1
#include <bits/stdc++.h>
vector<int> smallestSubarrayWithKDistinct(vector<int> &arr, int k)
{
    // Write your code here.
    int n=arr.size();
    int st=0,en=n;
    int i=0,j=0;
    unordered_map<int,int> um;
    while(j<n)
    {
        um[arr[j]]++;
        j++;
        while(um.size()==k)
        {
            if(j-i < en-st+1)//j-i-1 < en-st
            {
                st=i;
                en=j-1;
            }
            
            if(um[arr[i]]==1) um.erase(arr[i]);
            else um[arr[i]]--;
            i++;
        }
    }
    if(en==n) return {-1};
    return {st,en};
}

//57 three pointers
//a1 
#include <bits/stdc++.h> 

int threePointer(vector<int>& X, vector<int>& Y, vector<int>& Z)
{   
    // Write your code here.
    int a=X.size(),b=Y.size(),c=Z.size();
    int x=0,y=0,z=0;
    int p,q,r;
    int ans=INT_MAX;
    while(x<a && y<b && z<c)
    {
        p=X[x];q=Y[y];r=Z[z];
        if(p==q && p==r) return 0;
        if(p==q)
        {
            if(p<r) 
            {
                ans=min(ans,r-p);
                x++;
                y++;
            }
            else 
            {
                ans=min(ans,p-r);
                z++;
            }
        }
        else if(p==r)
        {
            if(p<q) 
            {
                ans=min(ans,q-p);
                x++;
                z++;
            }
            else 
            {
                ans=min(ans,p-q);
                y++;
            }
        }
        else if(q==r)
        {
            if(q<p) 
            {
                ans=min(ans,p-q);
                y++;
                z++;
            }
            else 
            {
                ans=min(ans,q-p);
                x++;
            }
        }
        else if(p<q)
        {
            if(p<r)
            {
                ans=min(ans,max(q-p,r-p));
                x++;
            }
            else 
            {
                ans=min(ans,max(p-r,q-r));
                z++;
            }
        }
        else 
            if(q<r)
            {
                ans=min(ans,max(p-q,r-q));
                y++;
            }
            else 
            {
                ans=min(ans,max(p-r,q-r));
                z++;
            }
    }
    return ans;
}

//a2 better
#include <bits/stdc++.h> 

int threePointer(vector<int>& X, vector<int>& Y, vector<int>& Z)
{   
    // Write your code here.
    int a=X.size(),b=Y.size(),c=Z.size();
    int x=0,y=0,z=0;
    int p,q,r;
    int ans=INT_MAX;
    while(x<a && y<b && z<c)
    {
        p=X[x];q=Y[y];r=Z[z];
        if(p==q && p==r) return 0;
        ans=min(ans,max({abs(p-q),abs(p-r),abs(q-r)}));
        if(p<=q && p<=r) x++;
        else if(q<=p and q<=r) y++;//and instead of &&
        else z++;
    }
    return ans;
}

//a3 select 3 elements 1 from each array such that maximum diff is minimum
#include <bits/stdc++.h> 

int threePointer(vector<int>& X, vector<int>& Y, vector<int>& Z)
{   
    // Write your code here.
    int a=X.size(),b=Y.size(),c=Z.size();
    int x=0,y=0,z=0;
    int p,q,r;
    int ans=INT_MAX;
    while(x<a && y<b && z<c)
    {
        p=X[x];
        q=Y[y];
        r=Z[z];
        if(p==q && p==r) return 0;
        int mi=min({p,q,r});
        int ma=max({p,q,r});
        ans=min(ans,ma-mi);
        if(p==mi) x++;
        else if(q==mi) y++;
        else z++;
    }
    return ans;
}

//58 Longest switching subarray
#include <bits/stdc++.h> 
int switchingSubarray(vector<int> &arr, int n) {
    // Write your code here.
    if(n==1) return n;
    int ans=2;
    int i=0,j=2;
    while(j<n)
    {
        while(j<n && arr[j]==arr[j-2])
        {           
            j++;            
        }
        ans=max(ans,j-i);
        i=j-1;
        j++;
    }
    return ans;
}

//a2
#include <bits/stdc++.h> 
int switchingSubarray(vector<int> &arr, int n) {
    // Write your code here.
    if(n==1) return 1;
    int ans=2;
    int cans=2;
    for(int i=0;i<n-2;i++)
    {
        if(arr[i+2]==arr[i]) 
        {
            cans++;
        }
        else 
        {
            ans=max(ans,cans);
            cans=2;
        }
    }
    ans=max(ans,cans);
    return ans;
}

//59 Operator precedence
1   ::  Scope resolution    Left-to-right →
2   a++   a--   Suffix/postfix increment and decrement
type()   type{} Functional cast
a() Function call
a[] Subscript
.   ->  Member access
3   ++a   --a   Prefix increment and decrement  Right-to-left ←
+a   -a Unary plus and minus
!   ~   Logical NOT and bitwise NOT
(type)  C-style cast
*a  Indirection (dereference)
&a  Address-of
sizeof  Size-of[note 1]
co_await    await-expression (C++20)
new   new[] Dynamic memory allocation
delete   delete[]   Dynamic memory deallocation
4   .*   ->*    Pointer-to-member   Left-to-right →
5   a*b   a/b   a%b Multiplication, division, and remainder
6   a+b   a-b   Addition and subtraction
7   <<   >> Bitwise left shift and right shift
8   <=> Three-way comparison operator (since C++20)
9   <   <=   >   >= For relational operators < and ≤ and > and ≥ respectively
10  ==   != For equality operators = and ≠ respectively
11  a&b Bitwise AND
12  ^   Bitwise XOR (exclusive or)
13  |   Bitwise OR (inclusive or)
14  &&  Logical AND
15  ||  Logical OR
16  a?b:c   Ternary conditional[note 2] Right-to-left ←
throw   throw operator
co_yield    yield-expression (C++20)
=   Direct assignment (provided by default for C++ classes)
+=   -= Compound assignment by sum and difference
*=   /=   %=    Compound assignment by product, quotient, and remainder
<<=   >>=   Compound assignment by bitwise left shift and right shift
&=   ^=   |=    Compound assignment by bitwise AND, XOR, and OR
17  ,   Comma   Left-to-right →

//60 equilibrium index
#include <bits/stdc++.h> 
vector<int> findEquilibriumIndices(vector<int> &sequence) {
    // Write your code here.
    int n=sequence.size();
    vector<int> l(n);
    vector<int> r(n);
    
    int s=0;
    for(int i=1;i<n;i++)
    {
        s+=sequence[i-1];
        l[i]=s;
    }
    
    s=0;
    for(int i=n-2;i>=0;i--)
    {
        s+=sequence[i+1];
        r[i]=s;
    }
    vector<int> ans;
    for(int i=0;i<n;i++)
    {
        if(l[i]==r[i]) ans.push_back(i);
    }
    return ans;
}

//a2
#include <bits/stdc++.h> 
vector<int> findEquilibriumIndices(vector<int> &sequence) {
    // Write your code here.
    int n=sequence.size();
    vector<int> pre(n);
    int s=0;
    for(int i=0;i<n;i++)
    {
        s+=sequence[i];
        pre[i]=s;
    }
    //at end s=sum of all
    vector<int> ans;
    for(int i=0;i<n;i++)
    {
        if(pre[i]-sequence[i] == s-pre[i]) ans.push_back(i);
    }
    return ans;
}

//61 Count Distinct Element in Every K Size Window
#include <bits/stdc++.h> 
vector<int> countDistinctElements(vector<int> &arr, int k) 
{
    // Write your code here
    if(k==1)
    {
        //cout<<"base";
        vector<int> A(arr.size(),1);
        return A;
    }
    
    unordered_map<int,int> um;
    int i=0,j=0,n=arr.size();
    while(j<k)//1<=k<=n
    {
        um[arr[j]]++;
        j++;
    }
    vector<int> ans;
    ans.push_back(um.size());
    
    while(j<n)
    {
        
        um[arr[j]]++;
        if(um[arr[i]]==1) um.erase(arr[i]);
        else um[arr[i]]--;
        //cout<<um.size()<<" ";
        
        ans.push_back(um.size());
        i++;
        j++;
    }
    //cout<<endl;
    
    return ans;
}


//62 check duplicate element present at distance of k
//sc O(n)
#include <bits/stdc++.h> 
bool checkDuplicate(vector<int> &arr, int n, int k) {
    // Write your code here.
    unordered_map<int,int> um;//1 based idx so 0 bydefault means element absent
    //val vs idx
    for(int i=0;i<n;i++)
    {
        if(um[arr[i]]>0)//0 absent , >0 present
        {
            if(i+1-um[arr[i]]<=k) return true;//check how far is latest idx of same element            
        }
        um[arr[i]]=i+1;//for 0 put 1 so >0 for all val
    }
    return false;
}

//a2 sc O(k)
#include <bits/stdc++.h> 
bool checkDuplicate(vector<int> &arr, int n, int k) {
    // Write your code here.
    unordered_set<int> us;
    for(int i=0;i<n;i++)
    {
        if(us.count(arr[i])) return true;
        us.insert(arr[i]);
        if(i>=k) us.erase(arr[i-k]);
    }
    return false;
}

//63 Longest Substring Without Repeating Characters
//n n 
#include <bits/stdc++.h> 
int lengthOfLongestSubstring(string &s) {
    // Write your code here.
    int i=0,j=0;
    int ans=1;
    int n=s.size();
    if(n==0 || n==1) return n;
    unordered_set<char> us;
    
    while(j<n)
    {
        while(j<n && us.count(s[j])==0)
        {
            us.insert(s[j]);           
            j++;            
        }
        ans=max(ans,j-i);
        if(j==n) break;
        while(us.count(s[j]))
        {
            us.erase(s[i]);
            i++;
        }
        
    }
    return ans;
}

//a2 faster
#include <bits/stdc++.h> 
int lengthOfLongestSubstring(string &s) {
    // Write your code here.
    int n=s.size();
    if(n==0 || n==1) return n;
    
    vector<int> v(256,-1);
    int st=-1,maxl=1;
    
    for(int i=0;i<n;i++)
    {
        if(v[s[i]]>st) st=v[s[i]];
        
        v[s[i]]=i;
        
        maxl=max(maxl,i-st);
    }
    return maxl;
}

//64 Fruits and Baskets
#include <bits/stdc++.h> 
int findMaxFruits(string &str, int n) {
    // Write your code here
    if(n<=2) return n;
    int i=0,j=0,ans=1;
    unordered_map<char,int> um;
    while(j<n)
    {
        um[str[j]]++;
        while(um.size()>2)
        {
            if(um[str[i]]==1) um.erase(str[i]);
            else um[str[i]]--;
            i++;
        }
        j++;
        ans=max(ans,j-i);
        
    }
    return ans;
    
}

//65 Anagram Substring Search
//
#include <bits/stdc++.h> 
vector<int> findAnagramsIndices(string str, string ptn, int n, int m)
{
    // Write you code here.
    unordered_map<char,int> um1;
    for(char c:ptn) um1[c]++;
    
    unordered_map<char,int> um2;
    int i=0,j=0;
    for( i=0;i<m;i++)
    {
        um2[str[i]]++;
    }
    
    vector<int> v;
    if(um1==um2) v.push_back(j);
    
    while(i<n)
    {
        um2[str[i]]++;
        if(um2[str[j]]==1) um2.erase(str[j]);
        else um2[str[j]]--;
        i++;
        j++;
        if(um1==um2) v.push_back(j);
        
        
    }
    return v;
    
}

//a2
#include <bits/stdc++.h> 

bool compare(vector<int> v1,vector<int> v2)
{
    for(int i=0;i<26;i++)
    {
        if(v1[i]!=v2[i]) return 0;
    }
    return 1;
}

vector<int> findAnagramsIndices(string str, string ptn, int n, int m)
{
    // Write you code here.
    vector<int> v1(26);
    vector<int> v2(26);
    
    for(int i=0;i<m;i++)
    {
        v1[ptn[i]-'A']++;
        v2[str[i]-'A']++;
    }
    vector<int> a;
    if(compare(v1,v2)) a.push_back(0);
    int i=0;
    for(int j=m;j<n;j++)
    {
        v2[str[j]-'A']++;
        v2[str[i]-'A']--;
        i++;
        if(compare(v1,v2)) a.push_back(i);
    }
    return a;
}

//a3
#include<bits/stdc++.h>
vector<int> findAnagramsIndices(string s, string ptn, int n, int m)
{
    // Write you code here.
    vector<int>ans;
    int k=m;  //size of window
    unordered_map<char,int>mp;
    for(int i=0;i<m;i++)
        mp[ptn[i]]++;
    
    int i=0,j=0;
    int count=mp.size();
    while(j<n)
    {  
        //calculation part
         if(mp.find(s[j])!=mp.end())
            {
                mp[s[j]]--;
                if(mp[s[j]]==0)
                    count--;   //decrease the size of distinct character.
            }
        if(j-i+1<k)    //window length not achieved so increase j
            j++;
        else if(j-i+1==k) //window size matches 
        {
           if(count==0)
               ans.push_back(i);
            if(mp.find(s[i])!=mp.end())
            {
                mp[s[i]]++;
                
                if(mp[s[i]]==1)
                    count++;
            }
            i++;   //slide the window;
            j++;
        }
    }
    return ans;
}