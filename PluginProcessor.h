#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

class DeEsserAudioProcessor : public juce::AudioProcessor {
public:
  DeEsserAudioProcessor();
  ~DeEsserAudioProcessor() override = default;

  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override {}
  bool isBusesLayoutSupported(const BusesLayout &layouts) const override;
  void processBlock(juce::AudioBuffer<float> &, juce::MidiBuffer &) override;

  juce::AudioProcessorEditor *createEditor() override;
  bool hasEditor() const override { return true; }

  const juce::String getName() const override { return "DeEsser"; }
  bool acceptsMidi() const override { return false; }
  bool producesMidi() const override { return false; }
  bool isMidiEffect() const override { return false; }
  double getTailLengthSeconds() const override { return 0.0; }
  int getNumPrograms() override { return 1; }
  int getCurrentProgram() override { return 0; }
  void setCurrentProgram(int) override {}
  const juce::String getProgramName(int) override { return {}; }
  void changeProgramName(int, const juce::String &) override {}

  void getStateInformation(juce::MemoryBlock &destData) override;
  void setStateInformation(const void *data, int sizeInBytes) override;

  float getAdaptiveFreq() const { return currentAdaptiveFreq; }

  using APVTS = juce::AudioProcessorValueTreeState;
  static APVTS::ParameterLayout createParameterLayout();
  APVTS apvts;

  // Scope
  struct ScopeFifo {
    juce::AbstractFifo fifo{8192};
    std::vector<float> data;
    int downsample = 256;
    int decimCount = 0;

    void prepare(int capacityPoints, double sr) {
      fifo.setTotalSize(capacityPoints);
      data.assign((size_t)capacityPoints * 3u, 0.0f);
      const int pointsPerSecond = 600;
      downsample =
          juce::jmax(1, (int)std::lround(sr / (double)pointsPerSecond));
      decimCount = 0;
      fifo.reset();
    }

    inline void pushSample(float wave, float suppress, float excite) {
      if (++decimCount < downsample)
        return;
      decimCount = 0;

      int start1, size1, start2, size2;
      fifo.prepareToWrite(1, start1, size1, start2, size2);
      if (size1 > 0)
        write(start1, wave, suppress, excite);
      else if (size2 > 0)
        write(start2, wave, suppress, excite);
      fifo.finishedWrite(1);
    }

    int pop(float *waveOut, float *suppressOut, float *exciteOut,
            int maxPoints) {
      const int toRead = juce::jmin(maxPoints, fifo.getNumReady());
      int start1, size1, start2, size2;
      fifo.prepareToRead(toRead, start1, size1, start2, size2);
      if (size1 > 0)
        read(start1, size1, waveOut, suppressOut, exciteOut);
      if (size2 > 0)
        read(start2, size2, waveOut, suppressOut, exciteOut);
      fifo.finishedRead(toRead);
      return toRead;
    }

  private:
    inline void write(int offset, float w, float s, float e) {
      const int i = offset * 3;
      data[(size_t)i] = w;
      data[(size_t)i + 1] = s;
      data[(size_t)i + 2] = e;
    }
    inline void read(int offset, int count, float *&w, float *&s, float *&e) {
      for (int n = 0; n < count; ++n) {
        const int i = (offset + n) * 3;
        *w++ = data[(size_t)i];
        *s++ = data[(size_t)i + 1];
        *e++ = data[(size_t)i + 2];
      }
    }
  };

  ScopeFifo &getScope() noexcept { return scope; }

  // FFT Spectrum
  struct FftScope {
    static constexpr int fftOrder = 11; // 2048 points
    static constexpr int fftSize = 1 << fftOrder;
    static constexpr int scopeSize = 512;

    juce::dsp::FFT forwardFFT{fftOrder};
    // Blackman-Harris window to reduce spectral leakage, fake lows/highs
    juce::dsp::WindowingFunction<float> window{
        fftSize, juce::dsp::WindowingFunction<float>::blackmanHarris};

    juce::AbstractFifo fifo{4096};
    std::vector<float> fifoBuffer;
    std::vector<float> fftData;
    std::vector<float> lastScopeData;
    float decayRate = 0.75f;

    FftScope() {
      fifoBuffer.resize(4096, 0.0f);
      fftData.resize(fftSize * 2, 0.0f);
      lastScopeData.resize(scopeSize, -100.0f);
    }

    void pushSamples(const juce::AudioBuffer<float> &buffer) {
      if (buffer.getNumChannels() < 1)
        return;
      auto *channelData = buffer.getReadPointer(0);
      int numSamples = buffer.getNumSamples();

      int start1, size1, start2, size2;
      fifo.prepareToWrite(numSamples, start1, size1, start2, size2);

      if (size1 > 0)
        std::copy_n(channelData, size1, fifoBuffer.data() + start1);
      if (size2 > 0)
        std::copy_n(channelData + size1, size2, fifoBuffer.data() + start2);

      fifo.finishedWrite(numSamples);
    }

    bool process(float *outputData) {
      if (fifo.getNumReady() < fftSize)
        return false;

      std::vector<float> temp(fftSize * 2);
      int start1, size1, start2, size2;
      fifo.prepareToRead(fftSize, start1, size1, start2, size2);

      if (size1 > 0)
        std::copy_n(fifoBuffer.data() + start1, size1, temp.data());
      if (size2 > 0)
        std::copy_n(fifoBuffer.data() + start2, size2, temp.data() + size1);

      fifo.finishedRead(fftSize / 2);

      window.multiplyWithWindowingTable(temp.data(), fftSize);
      forwardFFT.performFrequencyOnlyForwardTransform(temp.data());

      for (int i = 0; i < scopeSize; ++i) {
        float normalizedX = (float)i / (float)scopeSize;
        float index = std::pow(normalizedX, 2.0f) * (fftSize / 2);
        int bin = juce::jlimit(2, fftSize / 2 - 1, (int)index);

        // Normalize by FFT Size.
        // Without this, magnitudes are 2000x too large, causing peaking.
        float level = temp[(size_t)bin] / (float)fftSize;

        float db = juce::Decibels::gainToDecibels(level);

        // Clamp silence floor
        if (db < -100.0f)
          db = -100.0f;

        if (db < lastScopeData[i]) {
          db = lastScopeData[i] * decayRate + db * (1.0f - decayRate);
        }
        lastScopeData[i] = db;
        outputData[i] = db;
      }
      return true;
    }
  };

  FftScope &getFft() { return fftScope; }

private:
  struct Detector {
    float env = 0.0f;
    float envFull = 0.0f;
  };

  struct CrossoverBranch {
    juce::dsp::StateVariableTPTFilter<float> f1, f2;
    void prepare(const juce::dsp::ProcessSpec &spec) {
      f1.prepare(spec);
      f2.prepare(spec);
      f1.setType(juce::dsp::StateVariableTPTFilterType::lowpass);
      f2.setType(juce::dsp::StateVariableTPTFilterType::lowpass);
    }
    void setType(juce::dsp::StateVariableTPTFilterType t) {
      f1.setType(t);
      f2.setType(t);
    }
    void setFreq(float f) {
      f1.setCutoffFrequency(f);
      f2.setCutoffFrequency(f);
    }
    float process(float x) {
      return f2.processSample(0, f1.processSample(0, x));
    }
    void reset() {
      f1.reset();
      f2.reset();
    }
  };

  std::vector<Detector> detectors;
  std::vector<juce::dsp::StateVariableTPTFilter<float>> detectorFilters;
  std::vector<juce::dsp::StateVariableTPTFilter<float>>
      parametricFilters; // For Parametric Mode
  std::vector<CrossoverBranch> loXovers, hiXovers;

public:
  // Accessor for Visualization
  const juce::dsp::StateVariableTPTFilter<float> &
  getDetectorFilter(int channel) const {
    if (channel >= 0 && channel < detectorFilters.size())
      return detectorFilters[channel];
    static juce::dsp::StateVariableTPTFilter<float> dummy;
    return dummy;
  }

private:
  juce::AudioBuffer<float> scBuf, loBuf, hiBuf;
  juce::HeapBlock<float> grBuffer, envSib, envFullBuf, exciteCtrl;

  int bufferCapacity = 0;
  int grCapacity = 0;
  int envCapacity = 0;
  int exciteCapacity = 0;

  juce::AudioBuffer<float> delayBuffer;
  int delayPos = 0;
  static constexpr int LOOKAHEAD_MS = 2;
  int lookaheadSamples = 0;

  ScopeFifo scope;
  FftScope fftScope;

  juce::SmoothedValue<float, juce::ValueSmoothingTypes::Linear> outGainLin;
  double sr = 44100.0;

  float lastCenter = 6000.0f, lastQ = 2.0f, lastSplit = 6000.0f;

  // Adaptive Logic Variables
  std::atomic<float> *pAutoFreq = nullptr;
  juce::LinearSmoothedValue<float> smoothFreq;
  float currentAdaptiveFreq = 6000.0f;
  float prevSample[2] = {0.0f, 0.0f};

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeEsserAudioProcessor)
};