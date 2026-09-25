export default function TopOrderOption({ checked, onChange, disabled = false }: {
  checked: boolean; onChange?: (value: boolean) => void; disabled?: boolean;
}) {
  return <label className="my-3 flex items-start gap-2 text-sm text-gray-700">
    <input type="checkbox" className="mt-1" checked={checked} disabled={disabled}
      onChange={event => onChange?.(event.target.checked)} />
    <span>Feste TOP-Reihenfolge erzwingen
      <span className="block text-xs text-gray-500">TOPs werden in Agenda-Reihenfolge behandelt. Unklare Übergänge bleiben zur manuellen Prüfung offen. Gilt für neue Zuordnungen.</span>
    </span>
  </label>;
}
