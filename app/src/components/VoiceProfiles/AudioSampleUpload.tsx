import { Mic, Pause, Play, Upload } from 'lucide-react';
import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { FormControl, FormItem, FormMessage } from '@/components/ui/form';
import { useToast } from '@/components/ui/use-toast';

// The dropped file's browser-reported MIME type is unreliable (some OS/file-
// manager combos, especially on Windows, don't tag common audio containers
// as `audio/*`), so a type check alone silently swallows valid drops. Fall
// back to the file extension -- same permissiveness the "Choose file" input
// already has, since it has no type check at all.
const AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.aiff', '.webm'];

function looksLikeAudio(file: File): boolean {
  if (file.type.startsWith('audio/')) return true;
  const lowerName = file.name.toLowerCase();
  return AUDIO_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
}

interface AudioSampleUploadProps {
  file: File | null | undefined;
  onFileChange: (file: File | undefined) => void;
  onTranscribe: () => void;
  onPlayPause: () => void;
  isPlaying: boolean;
  isValidating?: boolean;
  isTranscribing?: boolean;
  isDisabled?: boolean;
  fieldName: string;
}

export function AudioSampleUpload({
  file,
  onFileChange,
  onTranscribe,
  onPlayPause,
  isPlaying,
  isValidating = false,
  isTranscribing = false,
  isDisabled = false,
  fieldName,
}: AudioSampleUploadProps) {
  const { t } = useTranslation();
  const { toast } = useToast();
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <FormItem>
      <FormControl>
        <div className="flex flex-col gap-2">
          <input
            type="file"
            accept="audio/*"
            name={fieldName}
            ref={fileInputRef}
            onChange={(e) => {
              const selectedFile = e.target.files?.[0];
              if (selectedFile) {
                onFileChange(selectedFile);
              } else {
                onFileChange(undefined);
              }
            }}
            className="hidden"
          />
          <div
            role="button"
            tabIndex={0}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setIsDragging(false);
            }}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              const droppedFile = e.dataTransfer.files?.[0];
              if (!droppedFile) return;
              if (looksLikeAudio(droppedFile)) {
                onFileChange(droppedFile);
              } else {
                toast({
                  title: t('audioSample.dropRejectedTitle'),
                  description: t('audioSample.dropRejectedDescription'),
                  variant: 'destructive',
                });
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            className={`flex flex-col items-center justify-center gap-4 p-4 border-2 rounded-lg transition-colors min-h-[180px] ${
              file
                ? 'border-primary bg-primary/5'
                : isDragging
                  ? 'border-primary bg-primary/5'
                  : 'border-dashed border-muted-foreground/25 hover:border-muted-foreground/50'
            }`}
          >
            {!file ? (
              <>
                <Button
                  type="button"
                  size="lg"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center gap-2"
                >
                  <Upload className="h-5 w-5" />
                  {t('audioSample.chooseFile')}
                </Button>
                <p className="text-sm text-muted-foreground text-center">
                  {t('audioSample.uploadHint')}
                </p>
              </>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <Upload className="h-5 w-5 text-primary" />
                  <span className="font-medium">{t('audioSample.fileUploaded')}</span>
                </div>
                <p className="text-sm text-muted-foreground text-center">
                  {t('audioSample.fileLabel', { name: file.name })}
                </p>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    size="icon"
                    variant="outline"
                    onClick={onPlayPause}
                    disabled={isValidating}
                    aria-label={isPlaying ? t('audioSample.pause') : t('audioSample.play')}
                  >
                    {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={onTranscribe}
                    disabled={isTranscribing || isValidating || isDisabled}
                    className="flex items-center gap-2"
                  >
                    <Mic className="h-4 w-4" />
                    {isTranscribing ? t('audioSample.transcribing') : t('audioSample.transcribe')}
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                      onFileChange(undefined);
                      if (fileInputRef.current) {
                        fileInputRef.current.value = '';
                      }
                    }}
                  >
                    {t('audioSample.remove')}
                  </Button>
                </div>
              </>
            )}
          </div>
        </div>
      </FormControl>
      <FormMessage />
    </FormItem>
  );
}
