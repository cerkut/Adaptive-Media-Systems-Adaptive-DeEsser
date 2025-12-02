#include "PluginEditor.h"
#include <cmath>

// Waveform View

WaveformView::WaveformView(DeEsserAudioProcessor &p)
    : processor(p), fifo(p.getScope()) {
  setOpaque(true);
  startTimerHz(60);

  waveform.resize(bufferSize, 0.0f);
  suppress.resize(bufferSize, 0.0f);
  excite.resize(bufferSize, 0.0f);

  drawWave.resize(bufferSize);
  drawSuppress.resize(bufferSize);
  drawExcite.resize(bufferSize);

  displayWave.resize(bufferSize);
  displaySuppress.resize(bufferSize);
  displayExcite.resize(bufferSize);
}

void WaveformView::paint(juce::Graphics &g) {
  g.fillAll(juce::Colour::fromRGB(10, 12, 20));

  if (numValid <= 1)
    return;

  auto bounds = getLocalBounds().toFloat();
  const float midY = bounds.getCentreY();
  const float xStep = bounds.getWidth() / (float)(numValid - 1);
  const float yGain = 0.95f * (bounds.getHeight() * 0.5f);

  // Waveform Path
  juce::Path wavePath;
  wavePath.startNewSubPath(bounds.getX(), midY);

  for (int i = 0; i < numValid; ++i) {
    float val = juce::jlimit(0.f, 1.f, displayWave[(size_t)i]);
    if (std::isnan(val))
      val = 0.0f;
    wavePath.lineTo(bounds.getX() + xStep * (float)i, midY - val * yGain);
  }
  for (int i = numValid - 1; i >= 0; --i) {
    float val = juce::jlimit(0.f, 1.f, displayWave[(size_t)i]);
    if (std::isnan(val))
      val = 0.0f;
    wavePath.lineTo(bounds.getX() + xStep * (float)i, midY + val * yGain);
  }
  wavePath.closeSubPath();

  // Fill
  {
    juce::ColourGradient grad(juce::Colour::fromRGB(130, 100, 255), 0.0f,
                              midY - yGain, juce::Colour::fromRGB(100, 80, 200),
                              0.0f, midY + yGain, false);
    g.setGradientFill(grad);
    g.fillPath(wavePath);
  }

  // Overlay
  {
    juce::Graphics::ScopedSaveState ss(g);
    g.reduceClipRegion(wavePath);

    for (int i = 0; i < numValid - 1; ++i) {
      float s = displaySuppress[(size_t)i];
      float e = displayExcite[(size_t)i];

      if (std::isnan(s))
        s = 0.0f;
      if (std::isnan(e))
        e = 0.0f;
      if (s <= 0.01f && e <= 0.01f)
        continue;

      float x = bounds.getX() + xStep * (float)i;
      float w = xStep + 1.0f;

      if (s > 0.01f) {
        g.setColour(juce::Colour::fromRGB(255, 100, 100).withAlpha(0.7f * s));
        g.fillRect(x, 0.0f, w, bounds.getHeight());
      } else if (e > 0.01f) {
        g.setColour(juce::Colours::cyan.withAlpha(0.6f * e));
        g.fillRect(x, 0.0f, w, bounds.getHeight());
      }
    }
  }
  g.setColour(juce::Colours::white.withAlpha(0.3f));
  g.strokePath(wavePath, juce::PathStrokeType(1.0f));

  // Auto Frequency Tracker
  if (auto *param = processor.apvts.getRawParameterValue("autoFreq")) {
    if (*param > 0.5f) {
      auto strip = bounds.removeFromTop(16.0f); // Small strip
      g.setColour(juce::Colours::black.withAlpha(0.4f));
      g.fillRect(strip);

      float freq = processor.getAdaptiveFreq();
      float minF = 3000.0f;
      float maxF = 12000.0f;
      float normPos =
          (std::log(freq) - std::log(minF)) / (std::log(maxF) - std::log(minF));
      normPos = juce::jlimit(0.0f, 1.0f, normPos);
      float xPos = strip.getX() + normPos * strip.getWidth();

      g.setColour(juce::Colours::cyan);
      g.drawVerticalLine((int)xPos, strip.getY(), strip.getBottom());

      g.setColour(juce::Colours::white.withAlpha(0.8f));
      g.setFont(10.0f);
      g.drawText(juce::String((int)freq) + "Hz", (int)xPos + 4,
                 (int)strip.getY(), 50, (int)strip.getHeight(),
                 juce::Justification::centredLeft);
    }
  }
}

void WaveformView::timerCallback() {
  constexpr int chunk = 512;
  float tmpWave[chunk], tmpSuppress[chunk], tmpExcite[chunk];
  const int popped = fifo.pop(tmpWave, tmpSuppress, tmpExcite, chunk);
  if (popped <= 0)
    return;

  for (int i = 0; i < popped; ++i) {
    writePos = (writePos + 1) % bufferSize;
    waveform[(size_t)writePos] = tmpWave[i];
    suppress[(size_t)writePos] = tmpSuppress[i];
    excite[(size_t)writePos] = tmpExcite[i];
  }
  numValid = juce::jmin(bufferSize, numValid + popped);
  // Buffer shifting logic
  const int count = numValid;
  const int start = (writePos - count + 1 + bufferSize) % bufferSize;
  if (start + count <= bufferSize) {
    std::copy(waveform.begin() + start, waveform.begin() + start + count,
              drawWave.begin());
    std::copy(suppress.begin() + start, suppress.begin() + start + count,
              drawSuppress.begin());
    std::copy(excite.begin() + start, excite.begin() + start + count,
              drawExcite.begin());
  } else {
    const int first = bufferSize - start;
    std::copy(waveform.begin() + start, waveform.end(), drawWave.begin());
    std::copy(waveform.begin(), waveform.begin() + (count - first),
              drawWave.begin() + first);
    std::copy(suppress.begin() + start, suppress.end(), drawSuppress.begin());
    std::copy(suppress.begin(), suppress.begin() + (count - first),
              drawSuppress.begin() + first);
    std::copy(excite.begin() + start, excite.end(), drawExcite.begin());
    std::copy(excite.begin(), excite.begin() + (count - first),
              drawExcite.begin() + first);
  }
  displayWave = drawWave;
  displaySuppress = drawSuppress;
  displayExcite = drawExcite;
  repaint();
}

// Spectrum View

SpectrumView::SpectrumView(DeEsserAudioProcessor &p) : processor(p) {
  setOpaque(true);
  startTimerHz(60);
  fftPoints.resize(512, -100.0f);
  detectorCurve.resize(512, -100.0f);
}

void SpectrumView::paint(juce::Graphics &g) {
  g.fillAll(juce::Colour::fromRGB(14, 16,
                                  24)); // Slightly darker/different background

  // Grid Lines for Freq
  g.setColour(juce::Colours::white.withAlpha(0.05f));
  g.drawHorizontalLine(getHeight() / 2, 0.0f, (float)getWidth());
  g.drawVerticalLine(getWidth() / 2, 0.0f, (float)getHeight());

  if (!fftPoints.empty()) {
    juce::Path fftPath;
    auto b = getLocalBounds().toFloat();
    bool started = false;

    for (int i = 0; i < (int)fftPoints.size(); ++i) {
      float db = fftPoints[i];
      float normY = 1.0f - juce::jmap(db, -100.0f, 0.0f, 0.0f, 1.0f);
      normY = juce::jlimit(0.0f, 1.0f, normY);

      float x = b.getX() + ((float)i / (float)fftPoints.size()) * b.getWidth();
      float y = b.getY() + normY * b.getHeight();

      if (!started) {
        fftPath.startNewSubPath(x, y);
        started = true;
      } else {
        fftPath.lineTo(x, y);
      }
    }

    // Gradient Stroke
    juce::ColourGradient strokeGrad(juce::Colours::cyan.withAlpha(0.0f), 0.0f,
                                    0.0f, juce::Colours::cyan,
                                    (float)getWidth(), 0.0f, false);

    g.setGradientFill(strokeGrad);
    g.strokePath(fftPath, juce::PathStrokeType(1.5f));

    // Subtle Fill
    fftPath.lineTo(b.getRight(), b.getBottom());
    fftPath.lineTo(b.getX(), b.getBottom());
    fftPath.closeSubPath();

    g.setColour(juce::Colours::cyan.withAlpha(0.1f));
    g.fillPath(fftPath);
  }

  // Detector Curve
  if (!detectorCurve.empty()) {
    juce::Path detPath;
    auto b = getLocalBounds().toFloat();
    bool started = false;

    for (int i = 0; i < (int)detectorCurve.size(); ++i) {
      float db = detectorCurve[i];
      float normY = 1.0f - juce::jmap(db, -100.0f, 0.0f, 0.0f, 1.0f);
      normY = juce::jlimit(0.0f, 1.0f, normY);

      float x =
          b.getX() + ((float)i / (float)detectorCurve.size()) * b.getWidth();
      float y = b.getY() + normY * b.getHeight();

      if (!started) {
        detPath.startNewSubPath(x, y);
        started = true;
      } else {
        detPath.lineTo(x, y);
      }
    }
    g.setColour(juce::Colours::yellow.withAlpha(0.6f));
    g.strokePath(detPath, juce::PathStrokeType(2.0f));
  }
}

void SpectrumView::timerCallback() {
  // Update FFT
  float tmpFft[512];
  if (processor.getFft().process(tmpFft)) {
    std::copy(std::begin(tmpFft), std::end(tmpFft), fftPoints.begin());
    repaint();
  }

  // Update Detector Curve
  // Calculate magnitude response of the detector filter
  auto &filter = processor.getDetectorFilter(0); // Ch 0 for display
  float freq = filter.getCutoffFrequency();
  float q = filter.getResonance();
  double sr = processor.getSampleRate();
  if (sr <= 0.0)
    sr = 44100.0;

  // Create coefficients for calculation 
  // (approximate SVF Bandpass)
  auto coeffs = juce::dsp::IIR::Coefficients<float>::makeBandPass(sr, freq, q);

  const int n = (int)detectorCurve.size();
  for (int i = 0; i < n; ++i) {
    // Map index to Frequency (matching FFT mapping)
    // Freq = (normalizedX^2) * (SampleRate / 2)
    float normalizedX = (float)i / (float)n;
    float f = std::pow(normalizedX, 2.0f) * (float)(sr * 0.5);

    if (f < 20.0f)
      f = 20.0f;
    if (f > sr * 0.5f)
      f = (float)sr * 0.5f;

    float mag = coeffs->getMagnitudeForFrequency(f, sr);
    float db = juce::Decibels::gainToDecibels(mag);

    // Normalize for display (Bandpass peak is 0dB)
    // SVF Bandpass gain at center is 0dB (gain=1)
    detectorCurve[i] = db;
  }
}

// Editor Implementation

DeEsserAudioProcessorEditor::DeEsserAudioProcessorEditor(
    DeEsserAudioProcessor &p)
    : AudioProcessorEditor(&p), processor(p) {
  juce::LookAndFeel::setDefaultLookAndFeel(&lnf);
  auto &apvts = processor.apvts;

  auto cDynamics = juce::Colour::fromRGB(255, 150, 150);
  auto cFilter = juce::Colour::fromRGB(200, 200, 200);
  auto cTone = juce::Colour::fromRGB(100, 240, 255);

  auto setup = [&](juce::Slider &s, juce::Label &l,
                   std::unique_ptr<APVTS::SliderAttachment> &att,
                   const char *paramID, const char *name, const char *suff,
                   juce::Colour col) {
    addAndMakeVisible(s);
    s.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 14);
    s.setTextValueSuffix(suff);
    s.setColour(juce::Slider::rotarySliderFillColourId, col);
    att = std::make_unique<APVTS::SliderAttachment>(apvts, paramID, s);

    addAndMakeVisible(l);
    l.setText(name, juce::dontSendNotification);
    l.setJustificationType(juce::Justification::centred);
    l.setFont(juce::FontOptions(13.0f, juce::Font::bold));
  };

  setup(threshold, lblThreshold, aThreshold, "threshold", "Threshold", " dB",
        cDynamics);
  setup(amount, lblAmount, aAmount, "amount", "Ratio", " %", cDynamics);
  setup(attack, lblAttack, aAttack, "attack", "Attack", " ms", cDynamics);
  setup(release, lblRelease, aRelease, "release", "Release", " ms", cDynamics);

  setup(center, lblCenter, aCenter, "centerFreq", "Detection", " Hz", cFilter);
  setup(q, lblQ, aQ, "q", "Q Factor", "", cFilter);
  setup(split, lblSplit, aSplit, "splitFreq", "Split Freq", " Hz", cFilter);

  setup(exciteAmount, lblExciteAmount, aExciteAmount, "exciteAmount", "Exciter",
        " %", cTone);
  setup(exciteMix, lblExciteMix, aExciteMix, "exciteMix", "Excite Mix", " %",
        cTone);
  setup(suppressMix, lblSuppressMix, aSuppressMix, "suppressMix", "Supp. Mix",
        " %", cDynamics);
  setup(outGain, lblOutGain, aOut, "outputGain", "Output", " dB",
        juce::Colours::white);

  addAndMakeVisible(mode);
  addAndMakeVisible(lblMode);
  lblMode.setText("Mode", juce::dontSendNotification);
  lblMode.setJustificationType(juce::Justification::centredRight);
  lblMode.setFont(juce::FontOptions(14.0f));
  lblMode.setColour(juce::Label::textColourId,
                    juce::Colours::white.withAlpha(0.5f));
  aMode = std::make_unique<APVTS::ComboBoxAttachment>(apvts, "mode", mode);

  addAndMakeVisible(btnListen);
  btnListen.setButtonText("Listen");
  aListen =
      std::make_unique<APVTS::ButtonAttachment>(apvts, "listen", btnListen);

  addAndMakeVisible(btnAuto);
  btnAuto.setButtonText("Auto");
  btnAuto.setClickingTogglesState(true);
  aAuto = std::make_unique<APVTS::ButtonAttachment>(apvts, "autoFreq", btnAuto);

  addAndMakeVisible(titleLabel);
  titleLabel.setText("Adaptive Deesser by StefBil", juce::dontSendNotification);
  titleLabel.setFont(juce::FontOptions(20.0f, juce::Font::bold));
  titleLabel.setJustificationType(juce::Justification::centredLeft);
  titleLabel.setColour(juce::Label::textColourId,
                       juce::Colours::white.withAlpha(0.9f));

  scopeView = std::make_unique<WaveformView>(processor);
  addAndMakeVisible(*scopeView);

  // Init Spectrum
  spectrumView = std::make_unique<SpectrumView>(processor);
  addAndMakeVisible(*spectrumView);

  setSize(800, 600);
}

DeEsserAudioProcessorEditor::~DeEsserAudioProcessorEditor() {
  juce::LookAndFeel::setDefaultLookAndFeel(nullptr);
}

void DeEsserAudioProcessorEditor::paint(juce::Graphics &g) {
  g.fillAll(juce::Colour::fromRGB(18, 20, 28));

  g.setColour(juce::Colours::white.withAlpha(0.08f));
  auto r = getLocalBounds();
  r.removeFromTop(180);
  int w = r.getWidth() / 3;
  g.drawVerticalLine(w, (float)r.getY() + 10.0f, (float)r.getBottom() - 10.0f);
  g.drawVerticalLine(w * 2, (float)r.getY() + 10.0f,
                     (float)r.getBottom() - 10.0f);
}

void DeEsserAudioProcessorEditor::resized() {
  auto area = getLocalBounds().reduced(15);

  auto header = area.removeFromTop(30);
  auto modeArea = header.removeFromRight(220);
  btnListen.setBounds(modeArea.removeFromRight(60));
  modeArea.removeFromRight(10);
  lblMode.setBounds(modeArea.removeFromLeft(50));
  mode.setBounds(modeArea);
  titleLabel.setBounds(header);


  // Waveform
  scopeView->setBounds(area.removeFromTop(100));

  // Gap between
  area.removeFromTop(5);

  // Spectrum (Bottom part of visualizer)
  spectrumView->setBounds(area.removeFromTop(50));

  // Gap before controls
  area.removeFromTop(15);

  // Controls
  int groupW = area.getWidth() / 3;
  auto gDyn = area.removeFromLeft(groupW).reduced(5, 0);
  auto gFilt = area.removeFromLeft(groupW).reduced(5, 0);
  auto gTone = area.reduced(5, 0);

  auto placeControl = [](juce::Rectangle<int> slot, juce::Label &l,
                         juce::Slider &s) {
    l.setBounds(slot.removeFromTop(18));
    s.setBounds(slot);
  };

  {
    int h = gDyn.getHeight() / 2;
    auto row1 = gDyn.removeFromTop(h);
    placeControl(row1.removeFromLeft(row1.getWidth() / 2).reduced(2),
                 lblThreshold, threshold);
    placeControl(row1.reduced(2), lblAmount, amount);
    placeControl(gDyn.removeFromLeft(gDyn.getWidth() / 2).reduced(2), lblAttack,
                 attack);
    placeControl(gDyn.reduced(2), lblRelease, release);
  }
  {
    int h = gFilt.getHeight() / 2;
    auto row1 = gFilt.removeFromTop(h);
    auto autoArea = row1.removeFromRight(50).reduced(0, 20);
    btnAuto.setBounds(autoArea);
    placeControl(row1.reduced(10, 0), lblCenter, center);
    placeControl(gFilt.removeFromLeft(gFilt.getWidth() / 2).reduced(2), lblQ,
                 q);
    placeControl(gFilt.reduced(2), lblSplit, split);
  }
  {
    int h = gTone.getHeight() / 2;
    auto row1 = gTone.removeFromTop(h);
    placeControl(row1.removeFromLeft(row1.getWidth() / 2).reduced(2),
                 lblExciteAmount, exciteAmount);
    placeControl(row1.reduced(2), lblExciteMix, exciteMix);
    placeControl(gTone.removeFromLeft(gTone.getWidth() / 2).reduced(2),
                 lblSuppressMix, suppressMix);
    placeControl(gTone.reduced(2), lblOutGain, outGain);
  }
}