import type { LayoutProps } from '../types';

export default function Layout({
  children,
  wide = false,
  onSettingsClick,
  onHistoryClick,
  onNewSessionClick,
}: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 py-3 sm:px-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="flex items-center gap-3 text-xl font-semibold text-gray-800">
            <img
              src="/gremienpilot-icon.svg"
              alt=""
              aria-hidden="true"
              className="h-9 w-9"
            />
            GremienPilot
          </h1>
          <div className="flex flex-wrap items-center gap-2">
            {onNewSessionClick && (
              <button
                type="button"
                onClick={onNewSessionClick}
                className="rounded-lg px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100"
              >
                Neue Sitzung
              </button>
            )}
            {onHistoryClick && (
              <button
                type="button"
                onClick={onHistoryClick}
                className="rounded-lg px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100"
              >
                Verlauf
              </button>
            )}
            {onSettingsClick && (
            <button
              type="button"
              onClick={onSettingsClick}
              className="flex items-center gap-2 p-2 text-sm text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="KI-Einstellungen"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              KI-Einstellungen
            </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="min-w-0 flex-1 wrap-anywhere px-4 py-4 sm:px-6">
        <div className={wide ? 'min-w-0' : 'max-w-6xl mx-auto'}>{children}</div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 px-6 py-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col items-center gap-4">
            <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-4">
              <img
                src="/logos/logo_bmftr_de.png"
                alt="BMFTR Logo"
                className="h-10 object-contain"
              />
              <img
                src="/logos/logo_aisc_150dpi.png"
                alt="AISC Logo"
                className="h-10 object-contain"
              />
              <a href="https://keule.dev" target="_blank" rel="noopener noreferrer">
                <img
                  src="/logos/keule-icon.svg"
                  alt="Keule Logo"
                  className="h-10 object-contain"
                />
              </a>
              <img
                src="/logos/Logo-jpeg_4c.jpg"
                alt="Doberlug-Kirchhain Logo"
                className="h-12 object-contain"
              />
            </div>
            <p className="text-sm text-gray-500">
              Weiterentwickelt durch{' '}
              <a href="https://keule.dev" target="_blank" rel="noopener noreferrer" className="underline hover:text-gray-700">
                Keule
              </a>{' '}
              und der Stadt Doberlug-Kirchhain
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
