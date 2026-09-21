import type { StepIndicatorProps } from '../types';

const steps = [
  { id: 1, name: 'Start' },
  { id: 2, name: 'Prüfen' },
  { id: 3, name: 'Exportieren' },
];

export default function StepIndicator({ currentStep }: StepIndicatorProps) {
  return (
    <div className="mb-4">
      <div className="flex items-center justify-center">
        {steps.map((step, index) => (
          <div key={step.id} className="flex items-center">
            {/* Step circle */}
            <div className="flex flex-col items-center gap-2 sm:flex-row" aria-current={step.id === currentStep ? 'step' : undefined}>
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium transition-colors ${
                  step.id < currentStep
                    ? 'bg-green-500 text-white'
                    : step.id === currentStep
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-500'
                }`}
              >
                {step.id < currentStep ? (
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                ) : (
                  step.id
                )}
              </div>
              <span
                className={`text-sm font-medium ${
                  step.id === currentStep ? 'text-blue-600' : 'text-gray-500'
                }`}
              >
                {step.name}
              </span>
            </div>

            {/* Connector line */}
            {index < steps.length - 1 && (
              <div
                className={`w-6 sm:w-16 h-1 mx-2 sm:mx-4 rounded ${
                  step.id < currentStep ? 'bg-green-500' : 'bg-gray-200'
                }`}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
