/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useEffect, useRef } from 'react';
import { 
  Upload, 
  Play, 
  Activity, 
  BarChart3, 
  Map as MapIcon, 
  TrendingUp, 
  Table as TableIcon,
  ChevronDown,
  ChevronUp,
  Settings,
  HelpCircle,
  MessageSquare,
  Zap,
  Download,
  X
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const API_BASE = (import.meta as { env?: { VITE_API_BASE?: string } }).env?.VITE_API_BASE ?? 'http://localhost:8000';

type AnalysisResult = {
  coordinates: { id: number; x: string; y: string; width: string; height: string }[];
  trajectory: { id: number; points: { x: number; y: number }[] }[];
  feeding_stats: { id: number; total_frames: number; feeding_frames: number; feeding_events: number }[];
  speed: { frame: number; id: number; speed_cm_s: number }[];
  acceleration: { frame: number; id: number; acceleration_cm_s2: number }[];
  angle: { frame: number; id: number; turn_angle: number }[];
  zone_stats: { zone: string; fish_count: number; feeding_count: number; feeding_events?: number }[];
  zone_over_time?: { time: number; center: number; middle: number; edge: number }[];
  cumulative_feeding_events_over_time?: { time: number; cumulative: number }[];
  heatmap: { x: number; y: number; value: number }[];
  feeding_frequency: { time: number; frequency: number }[];
  frame_shape: number[] | null;
  output_video_base64?: string;
};

// --- Components ---

const Card = ({ title, children, className, icon: Icon }: { title?: string, children: React.ReactNode, className?: string, icon?: any }) => (
  <div className={cn("bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden flex flex-col", className)}>
    {title && (
      <div className="px-5 py-4 border-bottom border-slate-50 flex items-center justify-between bg-slate-50/30">
        <div className="flex items-center gap-2">
          {Icon && <Icon className="w-4 h-4 text-blue-600" />}
          <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
        </div>
        <div className="flex gap-1">
          <div className="w-2 h-2 rounded-full bg-slate-200" />
          <div className="w-2 h-2 rounded-full bg-slate-200" />
        </div>
      </div>
    )}
    <div className={cn("p-5 flex-1", !title && "p-0")}>
      {children}
    </div>
  </div>
);

const ActivityHeatmap = ({ data }: { data: { x: number; y: number; value: number }[] }) => {
  const maxVal = data.length ? Math.max(...data.map((d) => d.value)) : 1;
  return (
    <div className="relative w-full aspect-square max-w-[300px] mx-auto rounded-full overflow-hidden border-4 border-slate-100 shadow-inner bg-slate-50">
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-full h-full rounded-full border-2 border-blue-100/50 bg-blue-50/10 flex items-center justify-center">
          <div className="w-[66%] h-[66%] rounded-full border-2 border-blue-200/50 bg-blue-100/20 flex items-center justify-center">
            <div className="w-[33%] h-[33%] rounded-full border-2 border-blue-300/50 bg-blue-200/30 flex items-center justify-center">
              <span className="text-[10px] font-bold text-blue-600/60 uppercase tracking-wider">中心区</span>
            </div>
            <span className="absolute top-[22%] text-[10px] font-bold text-blue-500/60 uppercase tracking-wider">中间区</span>
          </div>
          <span className="absolute top-[8%] text-[10px] font-bold text-blue-400/60 uppercase tracking-wider">边缘区</span>
        </div>
      </div>
      <div className="absolute inset-0">
        {data.slice(0, 200).map((d, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 0.6, scale: 1 }}
            transition={{ delay: i * 0.005 }}
            className="absolute w-3 h-3 rounded-full blur-[3px]"
            style={{
              left: `${d.x}%`,
              top: `${d.y}%`,
              backgroundColor: (d.value / maxVal) > 0.85 ? '#ef4444' : (d.value / maxVal) > 0.6 ? '#f97316' : (d.value / maxVal) > 0.35 ? '#fbbf24' : '#60a5fa',
            }}
          />
        ))}
      </div>
      <div className="absolute bottom-4 left-0 right-0 text-center pointer-events-none">
        <p className="text-[10px] text-slate-400 font-bold flex items-center justify-center gap-1 bg-white/60 backdrop-blur-sm py-1 px-2 rounded-full w-fit mx-auto">
          <Activity className="w-3 h-3" /> 活动热力视图
        </p>
      </div>
    </div>
  );
};

function groupByFrame(
  arr: { frame: number }[],
  valueKey: string,
  outKey: string
): { time: number; [k: string]: number }[] {
  const byFrame: Record<number, number[]> = {};
  for (const row of arr) {
    const v = Number((row as Record<string, unknown>)[valueKey]);
    if (!byFrame[row.frame]) byFrame[row.frame] = [];
    byFrame[row.frame].push(v);
  }
  return Object.entries(byFrame)
    .map(([frame, vals]) => ({ time: Number(frame), [outKey]: vals.reduce((a, b) => a + b, 0) / vals.length }))
    .sort((a, b) => a.time - b.time);
}

export default function App() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minConsecutiveFrames, setMinConsecutiveFrames] = useState(3);
  const [analysisDurationSeconds, setAnalysisDurationSeconds] = useState<number | null>(null);
  const [outputVideoBlobUrl, setOutputVideoBlobUrl] = useState<string | null>(null);
  type StreamingChartState = {
    zone_over_time: { time: number; center: number; middle: number; edge: number }[];
    cumulative_feeding_events_over_time: { time: number; cumulative: number }[];
    feeding_frequency: { time: number; frequency: number }[];
    zone_stats: { zone: string; fish_count: number; feeding_count: number; feeding_events?: number }[];
    heatmap: { x: number; y: number; value: number }[];
    speed: { frame: number; id: number; speed_cm_s: number }[];
    acceleration: { frame: number; id: number; acceleration_cm_s2: number }[];
    angle: { frame: number; id: number; turn_angle: number }[];
    trajectory: { id: number; points: { x: number; y: number }[] }[];
    coordinates: { id: number; x: string; y: string; width: string; height: string }[];
    feeding_stats: { id: number; total_frames: number; feeding_frames: number; feeding_events: number }[];
  };
  const [streamingChartData, setStreamingChartData] = useState<StreamingChartState | null>(null);
  const streamingAccumulatorRef = useRef<StreamingChartState | null>(null);
  const outputVideoUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const b64 = analysisResult?.output_video_base64;
    if (!b64) {
      if (outputVideoUrlRef.current) {
        URL.revokeObjectURL(outputVideoUrlRef.current);
        outputVideoUrlRef.current = null;
      }
      setOutputVideoBlobUrl(null);
      return;
    }
    try {
      const bin = atob(b64);
      const arr = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      const blob = new Blob([arr], { type: 'video/mp4' });
      const url = URL.createObjectURL(blob);
      if (outputVideoUrlRef.current) URL.revokeObjectURL(outputVideoUrlRef.current);
      outputVideoUrlRef.current = url;
      setOutputVideoBlobUrl(url);
      return () => {
        if (outputVideoUrlRef.current) {
          URL.revokeObjectURL(outputVideoUrlRef.current);
          outputVideoUrlRef.current = null;
        }
      };
    } catch {
      setOutputVideoBlobUrl(null);
    }
  }, [analysisResult?.output_video_base64]);

  useEffect(() => {
    if (!videoFile) {
      if (videoPreviewUrl) {
        URL.revokeObjectURL(videoPreviewUrl);
        setVideoPreviewUrl(null);
      }
      return;
    }
    const url = URL.createObjectURL(videoFile);
    setVideoPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  const coordinates = loading && streamingChartData?.coordinates?.length ? streamingChartData.coordinates : (analysisResult?.coordinates ?? []);
  const trajectorySource = loading && streamingChartData?.trajectory?.length ? streamingChartData.trajectory : (analysisResult?.trajectory ?? []);
  const trajectoryFlat = useMemo(() => {
    if (!trajectorySource.length) return [];
    return trajectorySource.flatMap((t) => t.points.map((p) => ({ ...p, z: 10 })));
  }, [trajectorySource]);
  const n = Math.max(1, minConsecutiveFrames);
  const feedingStats = loading && streamingChartData?.feeding_stats?.length ? streamingChartData.feeding_stats : (analysisResult?.feeding_stats ?? []);
  /** 有流式数据时用流式（.length 判断），与之前可正确流式显示区域鱼体数量变化的写法一致 */
  const cumulativeFeeding = loading && streamingChartData?.cumulative_feeding_events_over_time?.length
    ? streamingChartData.cumulative_feeding_events_over_time
    : (analysisResult?.cumulative_feeding_events_over_time ?? []);
  const totalFeedingEventsChartData = useMemo(() => {
    if (!cumulativeFeeding.length) return [];
    const maxFrame = Math.max(...cumulativeFeeding.map((p) => p.time));
    const points: { time: number; cumulative: number }[] = [];
    let lastCumulative = 0;
    for (let b = 0; (b + 1) * n <= maxFrame + n; b++) {
      const endFrame = (b + 1) * n;
      const atEnd = cumulativeFeeding.filter((p) => p.time <= endFrame).pop();
      const cumulative = atEnd?.cumulative ?? lastCumulative;
      lastCumulative = cumulative;
      points.push({ time: b, cumulative });
    }
    return points;
  }, [cumulativeFeeding, n]);
  const regionLineDataRaw = loading && streamingChartData?.zone_over_time?.length
    ? streamingChartData.zone_over_time
    : (analysisResult?.zone_over_time ?? []);
  const regionLineData = useMemo(() => {
    if (!regionLineDataRaw.length) return [];
    const byBlock = new Map<number, { center: number[]; middle: number[]; edge: number[] }>();
    for (const p of regionLineDataRaw) {
      const b = Math.floor((p.time - 1) / n);
      if (!byBlock.has(b)) byBlock.set(b, { center: [], middle: [], edge: [] });
      const cur = byBlock.get(b)!;
      cur.center.push(p.center);
      cur.middle.push(p.middle);
      cur.edge.push(p.edge);
    }
    return Array.from(byBlock.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([block, cur]) => ({
        time: block,
        center: cur.center.reduce((a, v) => a + v, 0),
        middle: cur.middle.reduce((a, v) => a + v, 0),
        edge: cur.edge.reduce((a, v) => a + v, 0),
      }));
  }, [regionLineDataRaw, n]);
  const trackingTableData = useMemo(() => {
    type Row = { id: number; feeding_events: number; x?: string; y?: string; width?: string; height?: string };
    const byId = new Map<number, Row>();
    for (const r of feedingStats) {
      byId.set(r.id, { id: r.id, feeding_events: r.feeding_events });
    }
    for (const c of coordinates) {
      const row: Row = byId.get(c.id) ?? ({ id: c.id, feeding_events: 0 } as Row);
      row.x = c.x;
      row.y = c.y;
      row.width = c.width;
      row.height = c.height;
      byId.set(c.id, row);
    }
    return Array.from(byId.values()).sort((a, b) => a.id - b.id);
  }, [feedingStats, coordinates]);
  const heatmapSource = loading && streamingChartData?.heatmap?.length ? streamingChartData.heatmap : (analysisResult?.heatmap ?? []);
  const heatmapData = heatmapSource;
  const speedSource = loading && streamingChartData?.speed?.length ? streamingChartData.speed : (analysisResult?.speed ?? []);
  const speedChartData = useMemo(
    () => (speedSource.length ? groupByFrame(speedSource, 'speed_cm_s', 'speed') : []),
    [speedSource]
  );
  const accelSource = loading && streamingChartData?.acceleration?.length ? streamingChartData.acceleration : (analysisResult?.acceleration ?? []);
  const accelChartData = useMemo(
    () => (accelSource.length ? groupByFrame(accelSource, 'acceleration_cm_s2', 'acceleration') : []),
    [accelSource]
  );
  const angleSource = loading && streamingChartData?.angle?.length ? streamingChartData.angle : (analysisResult?.angle ?? []);
  const angleChartData = useMemo(
    () => (angleSource.length ? groupByFrame(angleSource, 'turn_angle', 'angle') : []),
    [angleSource]
  );
  const zoneStatsSource = loading && streamingChartData?.zone_stats?.length ? streamingChartData.zone_stats : (analysisResult?.zone_stats ?? []);
  const zoneBarData = useMemo(() => {
    if (!zoneStatsSource.length) return [{ name: '中心区', value: 0 }, { name: '中间区', value: 0 }, { name: '边缘区', value: 0 }];
    const names: Record<string, string> = { center: '中心区', middle: '中间区', edge: '边缘区' };
    return zoneStatsSource.map((z) => ({ name: names[z.zone] ?? z.zone, value: z.feeding_events ?? z.feeding_count ?? 0 }));
  }, [zoneStatsSource]);

  const runAnalysis = async () => {
    if (!videoFile) return;
    const startMs = Date.now();
    setLoading(true);
    setError(null);
    setProgress(null);
    setAnalysisDurationSeconds(null);
    const initialStreaming: StreamingChartState = {
      zone_over_time: [],
      cumulative_feeding_events_over_time: [],
      feeding_frequency: [],
      zone_stats: [],
      heatmap: [],
      speed: [],
      acceleration: [],
      angle: [],
      trajectory: [],
      coordinates: [],
      feeding_stats: [],
    };
    streamingAccumulatorRef.current = initialStreaming;
    setStreamingChartData(initialStreaming);
    try {
      const form = new FormData();
      form.append('video', videoFile);
      const res = await fetch(`${API_BASE}/analyze-stream?min_consecutive_frames=${n}`, { method: 'POST', body: form });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || res.statusText);
      }
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error('无响应流');
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line) as {
              type: string;
              current?: number;
              total?: number;
              data?: AnalysisResult | { zone_over_time?: { time: number; center: number; middle: number; edge: number }[]; cumulative_feeding_events_over_time?: { time: number; cumulative: number }[]; feeding_frequency?: { time: number; frequency: number }[] };
              detail?: string;
            };
            if (msg.type === 'progress' && msg.current != null && msg.total != null) {
              setProgress({ current: msg.current, total: msg.total });
            } else if (msg.type === 'partial' && msg.data && !('output_video_base64' in (msg.data as object))) {
              const d = msg.data as {
                zone_over_time?: { time: number; center: number; middle: number; edge: number }[];
                cumulative_feeding_events_over_time?: { time: number; cumulative: number }[];
                feeding_frequency?: { time: number; frequency: number }[];
                zone_stats?: { zone: string; fish_count: number; feeding_count: number; feeding_events?: number }[];
                heatmap?: { x: number; y: number; value: number }[];
                speed?: { frame: number; id: number; speed_cm_s: number }[];
                acceleration?: { frame: number; id: number; acceleration_cm_s2: number }[];
                angle?: { frame: number; id: number; turn_angle: number }[];
                trajectory?: { id: number; points: { x: number; y: number }[] }[];
                coordinates?: { id: number; x: string; y: string; width: string; height: string }[];
                feeding_stats?: { id: number; total_frames: number; feeding_frames: number; feeding_events: number }[];
              };
              const prev = streamingAccumulatorRef.current;
              if (prev) {
                const next: StreamingChartState = {
                  zone_over_time: [...prev.zone_over_time, ...(d.zone_over_time ?? [])],
                  cumulative_feeding_events_over_time: [...prev.cumulative_feeding_events_over_time, ...(d.cumulative_feeding_events_over_time ?? [])],
                  feeding_frequency: [...prev.feeding_frequency, ...(d.feeding_frequency ?? [])],
                  zone_stats: d.zone_stats ?? prev.zone_stats,
                  heatmap: d.heatmap ?? prev.heatmap,
                  speed: [...prev.speed, ...(d.speed ?? [])],
                  acceleration: [...prev.acceleration, ...(d.acceleration ?? [])],
                  angle: [...prev.angle, ...(d.angle ?? [])],
                  trajectory: d.trajectory ?? prev.trajectory,
                  coordinates: d.coordinates ?? prev.coordinates,
                  feeding_stats: d.feeding_stats ?? prev.feeding_stats,
                };
                streamingAccumulatorRef.current = next;
                setStreamingChartData(next);
              }
            } else if (msg.type === 'result' && msg.data) {
              setAnalysisResult(msg.data as AnalysisResult);
              setAnalysisDurationSeconds((Date.now() - startMs) / 1000);
              setStreamingChartData(null);
            } else if (msg.type === 'error' && msg.detail) {
              setError(msg.detail);
            }
          } catch (_) {}
        }
      }
      if (buffer.trim()) {
        try {
          const msg = JSON.parse(buffer) as { type: string; data?: AnalysisResult & Record<string, unknown>; detail?: string };
          if (msg.type === 'result' && msg.data) {
            setAnalysisResult(msg.data);
            setAnalysisDurationSeconds((Date.now() - startMs) / 1000);
            setStreamingChartData(null);
          } else if (msg.type === 'error' && msg.detail) setError(msg.detail);
        } catch (_) {}
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '分析失败');
    } finally {
      setLoading(false);
      setProgress(null);
      setStreamingChartData(null);
      streamingAccumulatorRef.current = null;
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans selection:bg-blue-100 selection:text-blue-900">
      {/* Header */}
      <header className="h-16 bg-white border-b border-slate-200 px-6 flex items-center justify-between sticky top-0 z-50 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
            <Zap className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-slate-800">鱼类跟踪与行为量化系统</h1>
            <p className="text-[10px] text-slate-400 font-medium uppercase tracking-widest">Fish Tracking & Behavior Quantification System</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className={cn(
            "flex items-center gap-2 px-3 py-1.5 rounded-full border",
            loading ? "bg-amber-50 border-amber-100" : analysisResult ? "bg-emerald-50 border-emerald-100" : "bg-slate-50 border-slate-100"
          )}>
            <div className={cn(
              "w-2 h-2 rounded-full",
              loading && "bg-amber-500 animate-pulse",
              analysisResult && !loading && "bg-emerald-500",
              !analysisResult && !loading && "bg-slate-400"
            )} />
            <span className={cn(
              "text-xs font-semibold",
              loading && "text-amber-700",
              analysisResult && !loading && "text-emerald-700",
              !analysisResult && !loading && "text-slate-500"
            )}>
              {loading ? '分析中…' : analysisResult ? '已分析' : '未分析'}
            </span>
            {analysisResult && !loading && analysisDurationSeconds != null && (
              <span className="text-xs text-slate-500 font-medium">
                耗时 {analysisDurationSeconds < 60 ? `${analysisDurationSeconds.toFixed(1)} 秒` : `${Math.floor(analysisDurationSeconds / 60)} 分 ${(analysisDurationSeconds % 60).toFixed(1)} 秒`}
              </span>
            )}
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <Settings className="w-5 h-5 cursor-pointer hover:text-blue-600 transition-colors" />
            <HelpCircle className="w-5 h-5 cursor-pointer hover:text-blue-600 transition-colors" />
            <div className="w-8 h-8 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-600 font-bold text-xs">
              JD
            </div>
          </div>
        </div>
      </header>

      <main className="p-6 max-w-[1600px] mx-auto space-y-6">
        {/* Top Section: Sidebar + Video + New Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* Sidebar Column */}
          <div className="lg:col-span-3 flex flex-col gap-6">
            <Card title="视频上传" icon={Upload}>
              <div className="relative border-2 border-dashed border-slate-200 rounded-2xl overflow-hidden bg-slate-900 min-h-[200px]">
                {videoPreviewUrl ? (
                  <>
                    <video
                      src={videoPreviewUrl}
                      className="w-full aspect-video min-h-[200px] object-contain bg-black"
                      controls
                      playsInline
                      preload="auto"
                    />
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setVideoFile(null);
                        setError(null);
                        if (fileInputRef.current) fileInputRef.current.value = '';
                      }}
                      className="absolute top-2 right-2 w-8 h-8 rounded-full bg-black/60 hover:bg-black/80 text-white flex items-center justify-center transition-colors"
                      aria-label="清除并重新上传"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </>
                ) : (
                  <label className="block p-6 min-h-[200px] flex flex-col items-center justify-center gap-4 hover:border-slate-300 cursor-pointer group">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="video/*,.mp4,.avi,.mov"
                      className="hidden"
                      onChange={(e) => { const f = e.target.files?.[0]; if (f) setVideoFile(f); setError(null); }}
                    />
                    <div className="w-12 h-12 bg-slate-700 rounded-full flex items-center justify-center group-hover:bg-slate-600 transition-colors">
                      <Upload className="w-6 h-6 text-slate-300 group-hover:text-white" />
                </div>
                    <p className="text-sm font-medium text-slate-400 group-hover:text-slate-300">点击或拖拽上传视频</p>
                    <p className="text-xs text-slate-500">支持 MP4, AVI, MOV 格式</p>
                  </label>
                )}
                </div>
              <div className="mt-3 flex items-center gap-2 flex-wrap">
                <label className="text-xs text-slate-600 font-medium whitespace-nowrap">摄食判断：连续</label>
                <input
                  type="number"
                  min={1}
                  max={30}
                  value={minConsecutiveFrames}
                  onChange={(e) => setMinConsecutiveFrames(Math.max(1, Math.min(30, parseInt(String(e.target.value), 10) || 1)))}
                  className="w-14 py-1.5 px-2 rounded-lg border border-slate-200 text-sm text-center font-mono"
                />
                <span className="text-xs text-slate-600">帧判断为摄食状态</span>
              </div>
              {error && <p className="mt-2 text-xs text-red-500">{error}</p>}
              <button
                onClick={runAnalysis}
                disabled={!videoFile || loading}
                className={cn(
                  "w-full mt-4 py-3 rounded-xl font-semibold text-sm flex items-center justify-center gap-2",
                  videoFile && !loading ? "bg-blue-600 text-white hover:bg-blue-700 cursor-pointer" : "bg-slate-100 text-slate-400 cursor-not-allowed"
                )}
              >
                {loading ? (
                  <>处理中…</>
                ) : (
                  <><Play className="w-4 h-4" /> 开始分析</>
                )}
              </button>
            </Card>

            <Card title="追踪摄食数据" icon={Activity} className="flex-1">
              <div className="overflow-x-auto overflow-y-auto max-h-[272px] rounded-lg border border-slate-100">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-white z-10">
                    <tr className="text-slate-400 border-b border-slate-100">
                      <th className="pb-2 pt-2 px-2 font-medium">鱼 ID</th>
                      <th className="pb-2 pt-2 px-2 font-medium">摄食次数</th>
                      <th className="pb-2 pt-2 px-2 font-medium">X</th>
                      <th className="pb-2 pt-2 px-2 font-medium">Y</th>
                      <th className="pb-2 pt-2 px-2 font-medium">宽</th>
                      <th className="pb-2 pt-2 px-2 font-medium">高</th>
                    </tr>
                  </thead>
                  <tbody className="text-slate-600">
                    {trackingTableData.length > 0
                      ? trackingTableData.map((row, i) => (
                          <tr key={i} className="border-b border-slate-50 last:border-0">
                            <td className="py-2.5 px-2 font-bold text-blue-600">{row.id}</td>
                            <td className="py-2.5 px-2 font-mono">{row.feeding_events}</td>
                            <td className="py-2.5 px-2 font-mono">{row.x ?? '—'}</td>
                            <td className="py-2.5 px-2 font-mono">{row.y ?? '—'}</td>
                            <td className="py-2.5 px-2 font-mono">{row.width ?? '—'}</td>
                            <td className="py-2.5 px-2 font-mono">{row.height ?? '—'}</td>
                      </tr>
                        ))
                      : (
                        <tr><td colSpan={6} className="py-4 text-center text-slate-400 text-xs">上传视频并点击「开始分析」后显示</td></tr>
                      )}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          {/* Video Player Column (Middle) */}
          <div className="lg:col-span-6 flex flex-col">
            <Card title="预测视频" icon={Play} className="h-full">
              <div className="w-full min-h-[450px] bg-slate-900 rounded-xl flex flex-col items-center justify-center overflow-hidden">
                {analysisResult?.output_video_base64 && outputVideoBlobUrl ? (
                  <video
                    key={outputVideoBlobUrl}
                    src={outputVideoBlobUrl}
                    className="w-full min-h-[400px] object-contain bg-black rounded-xl"
                    controls
                    playsInline
                    preload="auto"
                  />
                ) : (
                  <div className="flex flex-col items-center gap-4 py-16">
                    <div className="w-16 h-16 bg-white/10 backdrop-blur-md rounded-full flex items-center justify-center border border-white/20">
                    <Play className="text-white w-8 h-8 fill-white" />
                    </div>
                    <p className="text-slate-400 text-sm font-medium text-center px-4">
                      {analysisResult ? '分析完成，图表已更新' : loading ? '正在分析视频…' : '请先上传视频并点击「开始分析」'}
                    </p>
                  </div>
                )}
              </div>
              {loading && (
                <div className="mt-3 space-y-1.5">
                  <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-blue-500 transition-all duration-300"
                      style={{
                        width: progress && progress.total > 0
                          ? `${Math.min(100, (100 * progress.current) / progress.total)}%`
                          : '30%',
                      }}
                    />
                </div>
                  <p className="text-xs text-slate-500 text-center">
                    {progress && progress.total > 0
                      ? `分析中 ${Math.round((100 * progress.current) / progress.total)}% (${progress.current}/${progress.total} 帧)`
                      : '正在加载模型…'}
                  </p>
                </div>
              )}
              {!loading && analysisResult?.output_video_base64 && outputVideoBlobUrl && (
                <div className="mt-3 flex justify-center">
                  <button
                    type="button"
                    onClick={() => {
                      const bin = atob(analysisResult!.output_video_base64!);
                      const arr = new Uint8Array(bin.length);
                      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
                      const blob = new Blob([arr], { type: 'video/mp4' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = '预测视频_鱼ID与摄食状态.mp4';
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                    className="flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
                  >
                    <Download className="w-4 h-4" /> 下载预测视频
                  </button>
                </div>
              )}
            </Card>
          </div>

          {/* New Charts Column (Far Right) */}
          <div className="lg:col-span-3 flex flex-col gap-6">
            <Card title="轨迹图" icon={MapIcon} className="flex-1">
              <div className="h-[200px] lg:h-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis type="number" dataKey="x" name="x" hide />
                    <YAxis type="number" dataKey="y" name="y" hide />
                    <ZAxis type="number" dataKey="z" range={[20, 100]} name="score" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Fish Trajectory" data={trajectoryFlat} fill="#3b82f6" line={{ stroke: '#3b82f6', strokeWidth: 1 }} shape={() => null} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card title="总计摄食次数随时间变化" icon={TrendingUp} className="flex-1">
              <div className="h-[200px] lg:h-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart key={`total-feeding-${loading}-${totalFeedingEventsChartData.length}`} data={totalFeedingEventsChartData} margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis dataKey="time" name="帧" hide />
                    <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <Tooltip 
                      contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                      formatter={(value: number) => [`${value} 次`, '累计摄食次数']}
                      labelFormatter={(label) => `帧 ${label}`}
                    />
                    <Line type="monotone" dataKey="cumulative" name="累计摄食次数" stroke="#3b82f6" strokeWidth={2} dot={false} activeDot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        </div>

        {/* Middle Section: Movement Parameters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card title="速度变化曲线 (cm/s)" icon={Activity}>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart key={`speed-${loading}-${speedChartData.length}`} data={speedChartData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                  <defs>
                    <linearGradient id="colorSpeed" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="time" hide />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  <Area type="monotone" dataKey="speed" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorSpeed)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card title="加速度变化曲线 (cm/s²)" icon={Activity}>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart key={`accel-${loading}-${accelChartData.length}`} data={accelChartData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                  <defs>
                    <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="time" hide />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  <Area type="monotone" dataKey="acceleration" stroke="#6366f1" strokeWidth={2} fillOpacity={1} fill="url(#colorAcc)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card title="转角变化曲线 (°)" icon={Activity}>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart key={`angle-${loading}-${angleChartData.length}`} data={angleChartData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                  <defs>
                    <linearGradient id="colorAngle" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.1}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="time" hide />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  <Area type="monotone" dataKey="angle" stroke="#8b5cf6" strokeWidth={2} fillOpacity={1} fill="url(#colorAngle)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        {/* Bottom Section: Spatial Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <Card title="鱼池活动热力图" icon={MapIcon} className="lg:col-span-4">
            <ActivityHeatmap data={heatmapData} />
          </Card>

          <Card title="区域鱼体数量变化曲线" icon={TrendingUp} className="lg:col-span-5">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart key={`region-${loading}-${regionLineData.length}`} data={regionLineData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="time" name="帧" hide />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  <Line type="monotone" dataKey="center" name="中心区" stroke="#3b82f6" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="middle" name="中间区" stroke="#60a5fa" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="edge" name="边缘区" stroke="#93c5fd" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-600" />
                <span className="text-xs text-slate-500 font-medium">中心区</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-400" />
                <span className="text-xs text-slate-500 font-medium">中间区</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-200" />
                <span className="text-xs text-slate-500 font-medium">边缘区</span>
              </div>
            </div>
          </Card>

          <Card title="各区域摄食次数统计图" icon={BarChart3} className="lg:col-span-3">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart layout="vertical" data={zoneBarData} margin={{ top: 10, right: 30, bottom: 0, left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" hide />
                  <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#64748b', fontWeight: 600 }} />
                  <Tooltip cursor={{ fill: '#f8fafc' }} contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    <Cell fill="#3b82f6" />
                    <Cell fill="#60a5fa" />
                    <Cell fill="#93c5fd" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 py-8 border-t border-slate-200 bg-white">
        <div className="max-w-[1600px] mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-400 font-medium">© 2026 鱼类行为量化系统 | 基于 YOLOv11 & BoTSORT</p>
          <div className="flex gap-8 text-xs font-semibold text-slate-500 uppercase tracking-widest">
            <a href="#" className="hover:text-blue-600 transition-colors">使用文档</a>
            <a href="#" className="hover:text-blue-600 transition-colors">API 接口</a>
            <a href="#" className="hover:text-blue-600 transition-colors">技术支持</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
