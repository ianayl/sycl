// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

// This test performs basic checks of parallel_for(nd_range, reduction, lambda)

#include "reduction_utils.hpp"

using namespace sycl;

struct AllIdOp {
  constexpr bool operator()(size_t Idx) const { return true; }
};

template <typename T, access::mode M> class MName;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t WGSize,
           size_t NWItems) {
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  range<1> GlobalRange = NDRange.get_global_range();
  buffer<T, 1> InBuf(GlobalRange);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  std::optional<T> CorrectOut;
  AllIdOp IdFilterFunc = {};

  // The value assigned here must be discarded (if IsReadWrite is true).
  // Verify that it is really discarded and assign some value.
  host_accessor(OutBuf, write_only)[0] = Init;

  //initInputData(InBuf, CorrectOut, BOp, GlobalRange, IdFilterFunc);
  size_t N = GlobalRange.size();
  host_accessor In_h(InBuf, write_only);
  for (int I = 0; I < N; ++I) {
    In_h[I] = ((I + 1) % 5) + 1.1;
    if (IdFilterFunc(I))
      ExpectedOut = ExpectedOut ? BOp(*ExpectedOut, In_h[I]) : In_h[I];
  }
  CorrectOut = CorrectOut ? BOp(*CorrectOut, Init) : Init;

  // Compute.
  Q.submit([&](handler &CGH) {
     // Helper for creating the reductions depending on the existance of an
     // identity.
     auto CreateReduction = [&]() {
        return reduction(ReduVarPtr, BOp, PropList);
     };

     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Redu = CreateReduction();
    CGH.parallel_for<Name>(Range, Redu, [=](nd_item<Dims> NDIt, auto &Sum) {
      if (IdFilterFunc(NDIt.get_global_linear_id()))
        Sum.combine(In[NDIt.get_global_linear_id()]);
    });
   }).wait();

  // Check correctness.
  host_accessor Out(OutBuf, read_only);
  T ComputedOut = *(Out.get_pointer());
  return checkResults(Q, BOp, Range, ComputedOut, *CorrectOut);

  //test<KName<Name, true>>(Q, Identity, Init, BOp, NDRange);
}

int main() {
  queue Q;
  printDeviceInfo(Q);
  tests<class A1, int>(
      Q, 0, 9, [](auto x, auto y) { return (x + y); }, 1, 1024);
  return 1;
}
