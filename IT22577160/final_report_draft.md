# CU Prescription and Recommendation Assessment

## Component Overview

This component of the AURA system focuses on CU Prescription and Recommendation Assessment for Chronic Spontaneous Urticaria (CSU) patients. The main objective of this module is to support clinical decision-making by recommending an appropriate treatment category and guideline step based on skin image evidence, laboratory findings, and patient clinical features. The component was designed as a multimodal intelligent system that combines deep learning, optical character recognition, explainable AI, and rule-based clinical guidance to generate a reliable and interpretable recommendation.

In real clinical practice, selecting an appropriate treatment pathway for CSU is often influenced by multiple factors such as visible skin manifestations, inflammatory and immunological laboratory values, symptom severity, and prior disease history. Manual assessment of these factors can be time-consuming and may vary from one clinician to another. Therefore, this component was developed to reduce assessment complexity and provide consistent AI-assisted recommendations that align with EAACI guideline-based treatment steps.

## Aim and Objectives

The aim of this component is to develop an AI-powered prescription recommendation module for CSU that can analyse patient skin images together with laboratory and clinical data and suggest a suitable drug group and treatment step.

The main objectives of the component are as follows:

1. To classify the most suitable drug recommendation group for a CSU patient.
2. To map the prediction to the corresponding EAACI guideline treatment step.
3. To extract important laboratory values automatically from uploaded lab report images using OCR.
4. To integrate symptom-based UAS7 severity scoring with AI predictions.
5. To provide explainable outputs using Grad-CAM and redness-based visual analysis.
6. To generate a structured clinical PDF report for decision-support purposes.

## Scope of the Component

The CU Prescription and Recommendation Assessment component accepts a skin image, relevant laboratory report images, and patient clinical details as inputs. The system then performs image processing, OCR-based text extraction, multimodal classification, severity interpretation, and explainability analysis before returning a treatment recommendation. The recommendation is not intended to replace a clinician. Instead, it acts as a decision-support tool that helps medical professionals make faster and more informed treatment decisions.

This component specifically supports prediction of four treatment-related drug groups:

1. `H1_ANTIHISTAMINE`
2. `LTRA`
3. `ADVANCED_THERAPY`
4. `OTHER`

In addition, the output is mapped into four CSU guideline treatment steps:

1. `STEP_1` - Standard-dose second-generation H1 antihistamines
2. `STEP_2` - Up-dosed H1 antihistamines
3. `STEP_3` - Omalizumab-based therapy
4. `STEP_4` - Ciclosporin or advanced immunosuppressive therapy

## System Architecture

The component was implemented as a FastAPI-based backend service within the `IT22577160` module. The core runtime loads a trained multimodal deep learning model and exposes endpoints for prediction, report generation, and data handoff.

The internal architecture of this component consists of the following major layers:

1. Input acquisition layer
2. OCR and preprocessing layer
3. Multimodal deep learning prediction layer
4. Severity scoring and guideline mapping layer
5. Explainability and morphological analysis layer
6. Clinical report generation layer

The input acquisition layer receives skin images, laboratory reports, and clinical form values. The OCR layer extracts values such as CRP, FT4, IgE, Vitamin D, and Age from report images. The prediction layer then combines image features, lab features, and clinical features through a multimodal neural network. After prediction, the system evaluates UAS7 severity, compares predicted treatment steps with guideline-aligned steps, and finally produces visual explanations and a PDF summary report.

## Methodology

### Multimodal Data Processing

This component was designed as a multimodal assessment pipeline because CSU treatment decisions cannot be derived from a single source of data. Skin appearance provides visual evidence of wheals and erythema, laboratory values reflect inflammatory and immunological status, and clinical details capture symptom severity and disease history. By combining these data sources, the component produces a more context-aware recommendation than an image-only classifier.

The laboratory features used by the model are:

1. CRP
2. FT4
3. IgE
4. Vitamin D
5. Age

The clinical features used by the model are:

1. Weight
2. Height
3. Age experienced first symptoms
4. Diagnosed at the age of
5. Itching score
6. Angioedema-related drug usage indicator

If laboratory values are not available through manual input or OCR extraction, the system uses training-set mean values as a fallback. This prevents extreme distortions in the feature space and ensures stable inference.

### OCR-Based Lab Report Extraction

The OCR module was implemented using `pytesseract` together with OpenCV preprocessing. Before text extraction, the uploaded report image is converted to grayscale, denoised using bilateral filtering, and enhanced using adaptive thresholding and morphological operations. This improves text visibility and helps the OCR engine recognize laboratory terms and numerical values more accurately.

The extracted report text is then matched against predefined laboratory synonyms to identify important biomarkers relevant to CSU decision support. This allows the component to work with differently formatted laboratory reports while still extracting the values required for prediction.

### Deep Learning Model

The main predictive model used in this component is `GC_MuPeN_v3`, a Gate-Controlled Multi-Path Ensemble Network based on EfficientNet-B3. The model was selected because the problem requires effective fusion of image, laboratory, and clinical modalities rather than relying only on visual cues.

The model architecture contains the following key elements:

1. EfficientNet-B3 image backbone for deep feature extraction from skin images
2. Separate MLP branches for laboratory and clinical feature processing
3. FiLM conditioning to modulate image features using laboratory embeddings
4. Gated fusion to dynamically weight image, lab, and clinical modalities
5. Dual output heads for drug group prediction and guideline step prediction

This architecture is beneficial because it allows the model to learn both modality-specific and cross-modal interactions. The gating mechanism is especially important because the influence of image, lab, and clinical data may vary from patient to patient.

### UAS7 Severity Interpretation

To support medically meaningful interpretation, the system uses UAS7 scoring as an additional severity indicator. UAS7 can be supplied directly or derived from daily wheal and pruritus averages. Once calculated, the system classifies disease severity into standard categories such as well-controlled, mild, moderate, or severe urticaria. The resulting severity interpretation is compared with the model-predicted treatment step to identify whether both decisions are aligned or whether the model suggests a lower or higher treatment step.

### Explainability and Image Characterisation

One important requirement of a medical AI system is interpretability. To address this, the component includes Grad-CAM visualisation and erythema-based image analysis. Grad-CAM highlights the image regions that most strongly influenced the predicted drug class. In addition, a redness heatmap is produced using the LAB colour space `a*` channel as an erythema proxy.

The component also computes CU-specific morphological characteristics, including:

1. Redness mean score
2. Redness coverage percentage
3. Erythema index
4. Wheal count
5. Average wheal diameter
6. Maximum wheal diameter
7. Wheal circularity
8. Aspect ratio
9. Distribution pattern
10. Shape description

These outputs improve transparency by giving clinicians a structured explanation of the visible disease pattern instead of only presenting a class label.

## Implementation Details

The component was implemented using Python and integrated into the backend using FastAPI. The main implementation files include the API endpoint layer, the multimodal model runtime, the OCR processing module, the explainability module, and the PDF report builder.

The implementation is organised around the following files:

1. `app/main.py` - API endpoints and end-to-end orchestration
2. `app/model_runtime.py` - model loading, preprocessing, inference, and guideline logic
3. `app/ocr_runtime.py` - OCR preprocessing and lab extraction
4. `app/explain.py` - Grad-CAM and CU image morphology analysis
5. `app/pdf_report.py` - clinical PDF report generation
6. `artifacts/config.json` - model feature configuration

The final API produces a structured JSON response that includes the predicted drug group, confidence score, top three predictions, mapped guideline step, feature values used for prediction, extracted labs, UAS7 interpretation, alignment status, and CU-specific visual characteristics.

## Results and Performance

The training summary available in the component artifacts indicates that the model was trained for 50 epochs using four drug classes and four treatment-step classes. The final training accuracy reached high values for both drug and step prediction, while the validation performance showed moderate generalisation.

The best observed validation metrics were as follows:

1. Best validation drug accuracy: `70.25%`
2. Best validation step accuracy: `72.11%`
3. Best validation loss: `1.3198`

The final epoch metrics were:

1. Training drug accuracy: `98.40%`
2. Validation drug accuracy: `68.18%`
3. Training step accuracy: `98.35%`
4. Validation step accuracy: `70.87%`

These results show that the model is capable of learning the relationship between multimodal patient data and treatment recommendations. However, the recorded training-validation gap suggests the presence of overfitting. Therefore, improvements such as stronger data augmentation, early stopping, higher regularisation, and broader dataset collection should be considered in future iterations.

## Strengths of the Component

This component provides several important strengths within the overall AURA system.

1. It combines image, laboratory, and clinical data in a single recommendation pipeline.
2. It automates extraction of laboratory values from report images.
3. It aligns predictions with recognized CSU treatment guideline steps.
4. It improves interpretability using Grad-CAM and redness-based visual analysis.
5. It generates a professional clinical PDF report for documentation and review.
6. It supports modular integration with the larger AURA backend and MongoDB persistence flow.

## Limitations

Although the component demonstrates promising performance, several limitations remain.

1. Validation accuracy is moderate and not yet sufficient for unsupervised clinical deployment.
2. Overfitting is visible in the final training report.
3. OCR performance may depend on report quality, resolution, and formatting.
4. The fallback mechanism for missing laboratory values may reduce patient-specific precision.
5. The current model is a decision-support system and should not be treated as a definitive prescribing authority.

## Future Improvements

Several enhancements can be proposed for future development of this component.

1. Expand the dataset with more clinically diverse CSU images and patient records.
2. Improve robustness through stronger augmentation and cross-validation strategies.
3. Fine-tune OCR extraction for local laboratory report formats.
4. Introduce confidence calibration and uncertainty-aware recommendations.
5. Add longitudinal tracking so repeated patient visits can be compared over time.
6. Integrate clinician feedback loops to refine recommendation quality.

## Conclusion

The CU Prescription and Recommendation Assessment component successfully demonstrates how multimodal artificial intelligence can support treatment planning for Chronic Spontaneous Urticaria. By combining skin image analysis, laboratory report extraction, clinical feature processing, guideline mapping, explainability, and report generation, the component provides a complete decision-support workflow for prescription assessment.

Overall, this module contributes significant practical value to the AURA research project because it transforms complex patient evidence into an interpretable recommendation that can assist clinicians during CSU management. Even though further optimisation is required to reduce overfitting and improve generalisation, the current implementation establishes a strong technical and clinical foundation for AI-assisted prescription recommendation in dermatological care.

## Suggested Short Version for Viva or Presentation

My component is the CU Prescription and Recommendation Assessment module of the AURA system. It is a multimodal AI component developed to recommend an appropriate treatment category and guideline step for Chronic Spontaneous Urticaria patients. The system accepts a skin image, laboratory report images, and clinical features as inputs. It uses OCR to extract important biomarkers, processes the skin image using an EfficientNet-B3 based multimodal model, evaluates UAS7 severity, generates explainability outputs such as Grad-CAM and redness maps, and finally produces a PDF-based clinical decision-support report. The model achieved around 70% validation accuracy for both drug-group prediction and treatment-step prediction, showing that the component has strong research potential while still requiring further improvement for real-world clinical use.
