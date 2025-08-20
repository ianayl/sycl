// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

// This test performs basic checks of parallel_for(nd_range, reduction, lambda)

#include <sycl/sycl.hpp>

using namespace sycl;

static constexpr size_t N = 1024;

int main() {
  queue q;

  auto BOp = [](int x, int y) { return (x + y); };

  // Ranges to be used:
  nd_range<1> NDRange(range<1>{N}, range<1>{1});
  range<1> GlobalRange = NDRange.get_global_range();

  // Initialize input, calculate correct output:
  int CorrectOut = 0;
  std::vector<int> In(N);
  for (int i = 0; i < N; ++i) {
    In[i] = ((i + 1) % 5) + 1;
    CorrectOut = BOp(CorrectOut, In[i]);
  }
  buffer<int, 1> InBuf(In.data(), GlobalRange);

  // Initialize output:
  int Out = 0;
  buffer<int, 1> OutBuf{&Out, 1};

  // Compute:
  q.submit([&](handler &CGH) {
     auto InAcc = InBuf.get_access<access_mode::read>(CGH);
     auto Redu = reduction(OutBuf, CGH, BOp);
     CGH.parallel_for<class Name>(NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
       Sum.combine(InAcc[NDIt.get_global_linear_id()]);
     });
   }).wait();

  // Check correctness.
  host_accessor ComputedOut{OutBuf, read_only};
  std::cout << ComputedOut[0] << " : " << CorrectOut << std::endl;

  return ComputedOut[0] != CorrectOut;
}
