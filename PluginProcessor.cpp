#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>

using namespace juce;

DeEsserAudioProcessor::DeEsserAudioProcessor()
    : AudioProcessor(
          BusesProperties()
              .withInput("Input", AudioChannelSet::stereo(), true)
              .withOutput("Output", AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMS", createParameterLayout()) {
  outGainLin.reset(0.0);
}

auto DeEsserAudioProcessor::createParameterLayout() -> APVTS::ParameterLayout {
  std::vector<std::unique_ptr<RangedAudioParameter>> p;

  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "threshold", "Threshold", NormalisableRange<float>(-60.f, 0.f, 0.01f),
      -18.f));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "amount", "Amount", NormalisableRange<float>(0.f, 100.f, 0.01f), 50.f));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "exciteAmount", "Excite Amount",
      NormalisableRange<float>(0.f, 100.f, 0.01f), 35.f));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "attack", "Attack", NormalisableRange<float>(0.05f, 20.f, 0.001f, 0.4f),
      2.f));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "release", "Release", NormalisableRange<float>(5.f, 300.f, 0.01f, 0.4f),
      80.f));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "centerFreq", "Detect Freq",
      NormalisableRange<float>(3000.f, 10000.f, 0.01f, 0.5f), 6000.f));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "q", "Detect Q", NormalisableRange<float>(0.4f, 5.f, 0.001f, 0.4f), 2.f));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "splitFreq", "Split Freq",
      NormalisableRange<float>(3000.f, 10000.f, 0.01f, 0.5f), 6000.f));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "suppressMix", "Suppress Mix",
      NormalisableRange<float>(0.f, 100.f, 0.01f), 100.f));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "exciteMix", "Excite Mix", NormalisableRange<float>(0.f, 100.f, 0.01f),
      40.f));
  p.emplace_back(std::make_unique<AudioParameterChoice>(
      "mode", "Mode",
      StringArray{"Shelf (Split)", "Wideband", "Bell (Parametric)"}, 2));
  p.emplace_back(
      std::make_unique<AudioParameterBool>("listen", "Listen", false));
  p.emplace_back(std::make_unique<AudioParameterFloat>(
      "outputGain", "Output Gain", NormalisableRange<float>(-24.f, 24.f, 0.01f),
      0.f));
  p.emplace_back(
      std::make_unique<AudioParameterBool>("autoFreq", "Auto Freq", false));

  return {p.begin(), p.end()};
}

bool DeEsserAudioProcessor::isBusesLayoutSupported(
    const BusesLayout &layouts) const {
  const auto in = layouts.getMainInputChannelSet();
  const auto out = layouts.getMainOutputChannelSet();
  return (in == out) &&
         (in == AudioChannelSet::mono() || in == AudioChannelSet::stereo());
}

void DeEsserAudioProcessor::prepareToPlay(double sampleRate,
                                          int samplesPerBlock) {
  sr = sampleRate;
  const int channels = jmax(1, getTotalNumInputChannels());

  dsp::ProcessSpec spec;
  spec.sampleRate = sampleRate;
  spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
  spec.numChannels = 1;

  detectors.resize(channels);
  for (auto &d : detectors) {
    d.env = 0.0f;
    d.envFull = 0.0f;
  }

  detectorFilters.resize(channels);
  parametricFilters.resize(channels);
  loXovers.resize(channels);
  hiXovers.resize(channels);

  for (auto &f : detectorFilters) {
    f.prepare(spec);
    f.setType(dsp::StateVariableTPTFilterType::bandpass);
    f.reset();
  }
  for (auto &f : parametricFilters) {
    f.prepare(spec);
    f.setType(dsp::StateVariableTPTFilterType::bandpass);
    f.reset();
  }
  for (auto &x : loXovers) {
    x.prepare(spec);
    x.setType(dsp::StateVariableTPTFilterType::lowpass);
    x.reset();
  }
  for (auto &x : hiXovers) {
    x.prepare(spec);
    x.setType(dsp::StateVariableTPTFilterType::highpass);
    x.reset();
  }

  bufferCapacity = samplesPerBlock;
  scBuf.setSize(channels, bufferCapacity);
  loBuf.setSize(channels, bufferCapacity);
  hiBuf.setSize(channels, bufferCapacity);

  grCapacity = bufferCapacity;
  grBuffer.allocate((size_t)grCapacity, true);
  envCapacity = bufferCapacity;
  envSib.allocate((size_t)envCapacity, true);
  envFullBuf.allocate((size_t)envCapacity, true);
  exciteCapacity = bufferCapacity;
  exciteCtrl.allocate((size_t)exciteCapacity, true);

  lookaheadSamples = static_cast<int>(LOOKAHEAD_MS * 0.001 * sr);
  delayBuffer.setSize(channels, lookaheadSamples + samplesPerBlock + 100);
  delayBuffer.clear();
  delayPos = 0;

  scope.prepare(8192, sampleRate);
  outGainLin.reset(sr, 0.05);
  outGainLin.setCurrentAndTargetValue(Decibels::decibelsToGain(
      apvts.getRawParameterValue("outputGain")->load()));

  // Initialize Adaptive Logic
  pAutoFreq = apvts.getRawParameterValue("autoFreq");
  smoothFreq.reset(sampleRate, 0.1);
  smoothFreq.setCurrentAndTargetValue(6000.0f);
  currentAdaptiveFreq = 6000.0f;
  prevSample[0] = 0.0f;
  prevSample[1] = 0.0f;
}

void DeEsserAudioProcessor::processBlock(AudioBuffer<float> &buffer,
                                         MidiBuffer &) {
  ScopedNoDenormals noDenormals;
  const int numCh = jmax(1, getTotalNumInputChannels());
  const int n = buffer.getNumSamples();

  // Resize Buffers if needed
  if (n > bufferCapacity) {
    bufferCapacity = n;
    scBuf.setSize(numCh, n);
    loBuf.setSize(numCh, n);
    hiBuf.setSize(numCh, n);
    if (n > grCapacity) {
      grCapacity = n;
      grBuffer.allocate((size_t)n, true);
    }
    if (n > envCapacity) {
      envCapacity = n;
      envSib.allocate((size_t)n, true);
      envFullBuf.allocate((size_t)n, true);
    }
    if (n > exciteCapacity) {
      exciteCapacity = n;
      exciteCtrl.allocate((size_t)n, true);
    }
  }

  // Clear Control Buffers
  float *gr = grBuffer.getData();
  float *exciteVals = exciteCtrl.getData();
  FloatVectorOperations::fill(gr, 1.0f, n);
  FloatVectorOperations::fill(exciteVals, 0.0f, n);

  // Load Parameters
  const float threshold = apvts.getRawParameterValue("threshold")->load();
  const float amountPct = apvts.getRawParameterValue("amount")->load();
  const float exciteAmountPct =
      apvts.getRawParameterValue("exciteAmount")->load();
  const float attackMs = apvts.getRawParameterValue("attack")->load();
  const float releaseMs = apvts.getRawParameterValue("release")->load();
  const float centerHz = apvts.getRawParameterValue("centerFreq")->load();
  const float q = apvts.getRawParameterValue("q")->load();
  const float splitHz = apvts.getRawParameterValue("splitFreq")->load();
  const float suppressMixPct =
      apvts.getRawParameterValue("suppressMix")->load();
  const float exciteMixPct = apvts.getRawParameterValue("exciteMix")->load();
  const int mode = (int)apvts.getRawParameterValue("mode")->load();
  const bool listen = (bool)*apvts.getRawParameterValue("listen");
  const float outDb = apvts.getRawParameterValue("outputGain")->load();
  const bool autoMode = (pAutoFreq && *pAutoFreq > 0.5f);

  // Update Auto-Frequency / Filters
  float targetF = autoMode ? currentAdaptiveFreq : centerHz;
  smoothFreq.setTargetValue(targetF);
  smoothFreq.skip(n);
  float nextFreq = smoothFreq.getCurrentValue();

  if (std::abs(nextFreq - lastCenter) > 1.0f || q != lastQ) {
    nextFreq = jlimit(3000.0f, 12000.0f, nextFreq);
    for (auto &f : detectorFilters) {
      f.setCutoffFrequency(nextFreq);
      f.setResonance(q);
    }
    for (auto &f : parametricFilters) {
      f.setCutoffFrequency(nextFreq);
      f.setResonance(q);
    }
    lastCenter = nextFreq;
    lastQ = q;
  }

  if (splitHz != lastSplit) {
    for (auto &x : loXovers)
      x.setFreq(splitHz);
    for (auto &x : hiXovers)
      x.setFreq(splitHz);
    lastSplit = splitHz;
  }
  outGainLin.setTargetValue(Decibels::decibelsToGain(outDb));

  // Prepare Envelopes
  const float atkA = std::exp(-1.0f / (0.001f * attackMs * (float)sr));
  const float relA = std::exp(-1.0f / (0.001f * releaseMs * (float)sr));
  const float amt = jlimit(0.0f, 1.0f, amountPct * 0.01f);
  const float exciteAmt = jlimit(0.0f, 1.0f, exciteAmountPct * 0.01f);

  for (int ch = 0; ch < numCh; ++ch)
    scBuf.copyFrom(ch, 0, buffer, ch, 0, n);

  float *envS = envSib.getData();
  float *envF = envFullBuf.getData();
  FloatVectorOperations::clear(envS, n);
  FloatVectorOperations::clear(envF, n);

  // Detection Loop
  for (int ch = 0; ch < numCh; ++ch) {
    auto *sc = scBuf.getWritePointer(ch);
    auto &df = detectorFilters[(size_t)ch];
    const float *in = buffer.getReadPointer(ch);

    float envS_ch = detectors[(size_t)ch].env;
    float envF_ch = detectors[(size_t)ch].envFull;

    for (int i = 0; i < n; ++i) {
      float bp = df.processSample(0, sc[i]);
      sc[i] = bp;

      float xs = std::abs(bp);
      envS_ch = xs + (xs > envS_ch ? atkA : relA) * (envS_ch - xs);
      if (envS_ch < 1.0e-9f)
        envS_ch = 0.0f;

      float xf = std::abs(in[i]);
      envF_ch = xf + (xf > envF_ch ? atkA : relA) * (envF_ch - xf);
      if (envF_ch < 1.0e-9f)
        envF_ch = 0.0f;

      envS[i] += envS_ch;
      envF[i] += envF_ch;
    }
    detectors[(size_t)ch].env = envS_ch;
    detectors[(size_t)ch].envFull = envF_ch;
  }

  // Calculate Gain Reduction
  const float invCh = 1.0f / (float)numCh;
  const float relThreshDb = threshold;
  const float threshRatio = std::pow(10.0f, relThreshDb * 0.05f);

  for (int i = 0; i < n; ++i) {
    float s = envS[i] * invCh;
    float f = envF[i] * invCh;
    const float minLvl = 1.0e-5f;

    if (f < minLvl)
      continue;

    if (s > f * threshRatio) {
      float overRatio = s / (f * threshRatio + 1.0e-9f);
      gr[i] = std::pow(overRatio, -amt);
      exciteVals[i] = 0.0f;
    } else {
      if (exciteAmt > 0.0f || exciteMixPct > 0.0f) {
        if (s > 1.0e-6f) {
          float ratioLin = s / (f + 1.0e-9f);
          float ratioDb = 20.0f * std::log10(ratioLin);
          float deficit = (relThreshDb + 20.0f) - ratioDb;
          exciteVals[i] = jlimit(0.0f, 1.0f, deficit / 12.0f);
        }
      }
    }
  }

  // Lookahead Delay
  int dW = delayPos;
  const int dLen = delayBuffer.getNumSamples();
  for (int ch = 0; ch < numCh; ++ch) {
    auto *src = buffer.getReadPointer(ch);
    auto *dst = delayBuffer.getWritePointer(ch);
    for (int i = 0; i < n; ++i) {
      dst[(dW + i) % dLen] = src[i];
    }
  }
  int dR = (delayPos - lookaheadSamples + dLen) % dLen;
  for (int ch = 0; ch < numCh; ++ch) {
    auto *dst = buffer.getWritePointer(ch);
    const auto *src = delayBuffer.getReadPointer(ch);
    for (int i = 0; i < n; ++i) {
      dst[i] = src[(dR + i) % dLen];
    }
  }
  delayPos = (delayPos + n) % dLen;

  // Handle Listen Mode
  if (listen) {
    for (int ch = 0; ch < numCh; ++ch) {
      FloatVectorOperations::copy(buffer.getWritePointer(ch),
                                  scBuf.getReadPointer(ch), n);
      FloatVectorOperations::multiply(buffer.getWritePointer(ch),
                                      outGainLin.getTargetValue(), n);
    }
    scope.pushSample(0.f, 0.f, 0.f);

    // Push Listen signal to FFT so spectrum works in Listen mode
    fftScope.pushSamples(buffer);
    return;
  }

  // Processing Application Split/Wideband
  const float suppressWet = suppressMixPct * 0.01f;
  const float exciteWetBase = exciteMixPct * 0.01f;
  double sumSig = 0.0;
  double sumDeriv = 0.0;

  for (int ch = 0; ch < numCh; ++ch) {
    const float *in = buffer.getReadPointer(ch);
    float *l = loBuf.getWritePointer(ch);
    float *h = hiBuf.getWritePointer(ch);
    auto *y = buffer.getWritePointer(ch);

    auto &lx = loXovers[(size_t)ch];
    auto &hx = hiXovers[(size_t)ch];

    for (int i = 0; i < n; ++i) {
      // Subtraction Crossover for Split-Band
      // Old: float loRaw = lx.process(in[i]); float hiRaw = hx.process(in[i]);
      // New: High = Input - Low.
      float loRaw = lx.process(in[i]);
      float hiRaw = in[i] - loRaw;

      // Parametric Mode Filter
      float paraBand = 0.0f;
      if (mode == 2) {
        paraBand = parametricFilters[(size_t)ch].processSample(0, in[i]);
      }

      l[i] = loRaw;
      h[i] = hiRaw;

      float g = gr[i];

      // Mode Processing
      float hiOut = hiRaw;
      float loOut = loRaw;

      if (mode == 2) // Parametric
      {
          // Dynamic EQ Suppression
          // Output = Input - (Bandpass * reduction)
          float reduction = (1.0f - g) * suppressWet;

          // Exciter Logic Ported to Parametric
          float exc = 0.0f;
          float ctrl = exciteVals[i];

          if (ctrl > 0.01f && exciteAmt > 0.01f) {
              float drive = 1.0f + 3.0f * exciteAmt * ctrl;
              float sat = std::tanh(drive * paraBand);
              float norm = std::tanh(drive);
              if (norm > 0.001f) sat /= norm;

              float wet = exciteWetBase * ctrl;
              float boost = std::pow(10.0f, (9.0f * exciteAmt * ctrl) * 0.05f);

              // Calculate added harmonic content
              exc = wet * (sat * boost - paraBand);
          }

          // Apply both Suppression and Excitation
          y[i] = in[i] - (paraBand * reduction) + exc;
      } else // Split-Band or Wideband
      {
        // High Band Suppress
        float hiSuppTarget = hiRaw * g;
        float hiSupp = hiRaw + suppressWet * (hiSuppTarget - hiRaw);
        hiOut = hiSupp;

        // Exciter Only High Band
        float ctrl = exciteVals[i];
        if (ctrl > 0.01f && exciteAmt > 0.01f) {
          float drive = 1.0f + 3.0f * exciteAmt * ctrl;
          float sat = std::tanh(drive * hiSupp);
          float norm = std::tanh(drive);
          if (norm > 0.001f)
            sat /= norm;

          float wet = exciteWetBase * ctrl;
          float boost = std::pow(10.0f, (9.0f * exciteAmt * ctrl) * 0.05f);
          hiOut = hiSupp + wet * (sat * boost - hiSupp);
        }

        // Low Band
        if (mode == 1) { // Wideband Mode
          float loTarget = loRaw * g;
          loOut = loRaw + suppressWet * (loTarget - loRaw);
        }

        y[i] = loOut + hiOut;
      }
    }

    // Adaptive Stats Improved High-Frequency Focus
    if (autoMode) {
        // Local state for 2nd derivative approximation
        float lastD1 = 0.0f;

        for (int i = 0; i < n; ++i) {
            float samp = h[i];

            // 1st Derivative (Velocity) - Emphasizes Highs (+6dB/oct)
            float d1 = samp - prevSample[ch];
            prevSample[ch] = samp;

            // 2nd Derivative (Acceleration) - For the "velocity" centroid
            float d2 = d1 - lastD1;
            lastD1 = d1;

            if (std::abs(d1) > 1.0e-5f) {
                // energy of the Derivative (d1) and 2nd Derivative (d2)
                sumSig += (double)(d1 * d1);
                sumDeriv += (double)(d2 * d2);
            }
        }
    }
  }

  // Update Adaptive Frequency
   // Gating lowered for derivative energy + Smoothing
  if (autoMode && sumSig > 0.00001) { // Threshold lowered because derivatives are smaller
      double ratio = std::sqrt(sumDeriv / sumSig);
      float instFreq = (float)((sr / 6.283185307) * ratio);

      // Clamp to reasonable sibilance range
      instFreq = jlimit(3000.0f, 11000.0f, instFreq);

      float diff = instFreq - currentAdaptiveFreq;

      // Slew Limiting Smoothing
      float maxChange = 50.0f;
      diff = jlimit(-maxChange, maxChange, diff);

      currentAdaptiveFreq += diff * 0.5f;
  }

  // Final Output Gain
  const float gVal = outGainLin.getNextValue();
  buffer.applyGain(gVal);

  // Push to Visualizers - PostFX
  {
    const float *yL = buffer.getReadPointer(0);
    const float *yR = (numCh > 1) ? buffer.getReadPointer(1) : nullptr;

    for (int i = 0; i < n; ++i) {
      float valL = std::abs(yL[i]);
      float valR = (yR) ? std::abs(yR[i]) : 0.0f;
      float wave = jmax(valL, valR);
      if (wave > 10.0f)
        wave = 1.0f;
      if (std::isnan(wave))
        wave = 0.0f;
      float gDisplay = 1.0f - gr[i];
      if (std::isnan(gDisplay))
        gDisplay = 0.0f;
      scope.pushSample(wave, gDisplay, exciteVals[i]);
    }
  }
  fftScope.pushSamples(buffer);
}

void DeEsserAudioProcessor::getStateInformation(juce::MemoryBlock &destData) {
  if (auto xml = apvts.copyState().createXml())
    copyXmlToBinary(*xml, destData);
}

void DeEsserAudioProcessor::setStateInformation(const void *data,
                                                int sizeInBytes) {
  if (auto xml = getXmlFromBinary(data, sizeInBytes))
    apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessorEditor *DeEsserAudioProcessor::createEditor() {
  return new DeEsserAudioProcessorEditor(*this);
}

juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter() {
  return new DeEsserAudioProcessor();
}